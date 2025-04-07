"""
This script contains the biomechanics-aware trajectory optimization module.
It takes care of the interface to our KUKA LBR iiwa 7 robot (acting as a robotic physiotherapist),
and implements a nonlinear programming problem (NLP) to find the trajectory that the robot needs to follow
for minimizing rotator cuff strain in a subject. This is done by embedding a biomechanical model of the
human shoulder (glenohumeral joint) in the optimization problem.

OpenSim, OpenSimAD and CasADi are used to streamline the computations, and the output optimal trajectories
are transformed in robot references that are published continuously to ROS topics, so they are sent to
the robot's controller.
"""

import opensim as osim
import casadi as ca
import numpy as np
import rospy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import time
import threading
import pygame
import ctypes

# Messages
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Bool
from geometry_msgs.msg import PoseStamped

from experiment_parameters import *     # this contains the experimental_params and the shared_ros_topics

from botasensone import BotaSerialSensor
import astar

class TO_module:
    """
    Trajectory Optimization module.
    """
    def __init__(self, nlps, shared_ros_topics, rate=200, with_opensim = False, simulation = 'true', speed_estimate:Bool=False, ft_sensor=None, rmr_solver=None):
        """"
        The TO_module requires a NLP object when instantiated
        """
        self.gravity_in_world = np.array([0, 0, -9.81])      # gravity vector expressed in the world frame 

        # set debug level
        self.simulation = simulation

        # Parameters
        self.with_opensim = with_opensim            # flag to indicate whether OpenSim could be imported (used mostly for visualization)

        self.strainmap_current = None               # strain map corresponding to the current model state

        self.strainmaps_params_dict = None          # Dictionary containing of all the precomputed (and interpolated) strain maps. 
                                                    # They are stored in memory as parameters of the corresponding Gaussians and 
                                                    # the correct set of parameters is selected at every time instant (and set to strain map_current)

        self.state_space_names = None               # The names of the coordinates in the model that describe the state-space in which we move
                                                    # For the case of our shoulder rehabilitation, it will be ["plane_elev", "shoulder_elev", "axial_rot"]

        self.state_values_current = None            # The current value of the variables defining the state
        self.state_dot_current = None               # The current value of the time derivatives of the state variables
        self.speed_estimate = speed_estimate        # Are estimated velocities of the model computed?
                                                    # False: quasi_static assumption, True: updated through measurements


        self.x_opt = None                           # Optimal trajectory in shoulder state. It consists in a sequence of points (or a single point)
                                                    # that will be transformed in robot coordinates and tracked by the robot.
        self.u_opt = None                           # Optimal sequence of generalized torques to be applied to the model to track
                                                    # the desired x_opt

        self.nlps_module = nlps                     # module for treating the NLP problem
        self.nlp_count = 0                          # keeps track of the amount of times the NLP is solved
        self.avg_nlp_time = 0
        self.failed_count = 0

        self.astar_planner = astar.Astar()         # A* planner for the strain map (as implemented in the RAL paper from Micah)

        # parameters regarding the muscle activation
        self.rmr_solver = rmr_solver                # instance of the RMR solver for estimating muscle activations in real time
        self.varying_activation = False             # initialize the module assuming that the activation will not change
        self.activation_level = 0                   # level of activation of the muscles whose strain we are considering
                                                    # this is an index 
        self.max_activation = 1                     # maximum activation considered (defaulted to 1)
        self.min_activation = 0                     # minimum activation considered (defaulted to 0)
        self.delta_activation = 0.005               # resolution on the activation value that the strain map captures

        self.rmr_thread = None                      # thread that will run the RMR solver


        # MPC optimal map generated with CasADi opti.to_function() for iteratively solving the same
        # NLP with different parameter values
        self.MPC_iter = None                        # CasADi function that maps initial state to the optimal control

        # current estimate of the pose of the robotic end-effector 
        self.current_ee_pose = None         # XYZ cartesian position in robot base frame
        self.current_ee_orientation = None  # quaternion orientation of the robot end-effector (scalar last)

        # filter the reference that is generated, to avoid noise injection from the human-pose estimator
        self.last_cart_ee_cmd = np.zeros(3)     # store the last command sent to the robot (position)
        self.last_rotation_ee_cmd = np.zeros(3)     # store the last command sent to the robot (rotation)
        self.alpha_p = 0.7                        # set the weight used in the exponential moving average filter
        self.alpha_r = 0.7                        # set the weight used in the exponential moving average filter
        self.filter_initialized = False         # has the filter been initialized already?

        # ROS part
        # initialize ROS node and set required frequency
        rospy.init_node('TO_module')
        self.ros_rate = rospy.Rate(rate)

        # Create publisher for the optimal cartesian trajectory for the KUKA end-effector
        self.topic_opt_traj = shared_ros_topics['optimal_cartesian_ref_ee']
        self.pub_trajectory = rospy.Publisher(self.topic_opt_traj, PoseStamped, queue_size=1)
        self.flag_pub_trajectory = False    # flag to check if trajectory is being published (default: False = no publishing)

        # create a publisher for the estimated muscle activations and for the compensated interaction forces
        self.pub_activation = rospy.Publisher(shared_ros_topics['muscle_activation'], Float32MultiArray, queue_size=1)
        self.pub_compensated_wrench = rospy.Publisher(shared_ros_topics['compensated_wrench'], Float32MultiArray, queue_size=1)
        
        # Create the publisher for the unused z_reference
        # It will publish the uncompensated z reference when running a real experiment,
        # and the gravity compensated reference when running in simulation.
        self.topic_z_level = shared_ros_topics['z_level']
        self.pub_z_level = rospy.Publisher(self.topic_z_level, Float32MultiArray, queue_size=1)

        # Create the publisher dedicated to stream the optimal trajectories and controls
        self.topic_optimization_output = shared_ros_topics['optimization_output']
        self.pub_optimization_output = rospy.Publisher(self.topic_optimization_output, Float32MultiArray, queue_size=1)

        # Create a subscriber to listen to the current value of the shoulder pose
        self.topic_shoulder_pose = shared_ros_topics['estimated_shoulder_pose']
        self.sub_curr_shoulder_pose = rospy.Subscriber(self.topic_shoulder_pose, Float32MultiArray, self._shoulder_pose_cb, queue_size=1)
        self.flag_receiving_shoulder_pose = False       # flag indicating whether the shoulder pose is being received

        # Set up the structure to deal with the new thread, to allow continuous publication of the optimal trajectory and torques
        # The thread itself is created later, with parameters known at run time
        self.x_opt_lock = threading.Lock()          # Lock for synchronizing access to self.x_opt
        self.publish_thread = None                  # creating the variable that will host the thread

        # create a subscriber to catch when the trajectory optimization should be running
        self.flag_run_optimization = False
        self.sub_run_optimization = rospy.Subscriber(shared_ros_topics['request_reference'], Bool, self._flag_run_optimization_cb, queue_size=1)

        self.time_begin_optimizing = None           # store the time at which the optimization request begins

        # Force-Torque sensor
        self.ft_sensor = ft_sensor                  # object to handle the force-torque sensor
        self.sensor_R_ee = experimental_params['sensor_R_ee']   # rotation matrix from the end effector frame to sensor frame

        if self.ft_sensor is not None:
            # specify the sensor load parameters (coming from the brace mounted on it)
            self.sensor_load_mass = experimental_params['brace_mass']   # mass of the load
            self.sensor_load_com = experimental_params['brace_com']     # center of mass of the load (in the sensor frame)

            # we calibrate when the sensor's frame is aligned with the robot's world frame
            # (we know precisely the effects of the load in this configuration)
            fx_load = self.sensor_load_mass * self.gravity_in_world[0]  # force in the x direction
            fy_load = self.sensor_load_mass * self.gravity_in_world[1]  # force in the y direction
            fz_load = self.sensor_load_mass * self.gravity_in_world[2]  # force in the z direction
            mx_load = fy_load * self.sensor_load_com[2] - fz_load * self.sensor_load_com[1]  # moment in the x direction
            my_load = fz_load * self.sensor_load_com[0] - fx_load * self.sensor_load_com[2]  # moment in the y direction
            mz_load = fx_load * self.sensor_load_com[1] - fy_load * self.sensor_load_com[0]  # moment in the z direction
            self.ft_sensor.set_load_effect(np.array((fx_load, fy_load, fz_load, mx_load, my_load, mz_load)))
            rospy.loginfo("ft load effect set to: %s", str((fx_load, fy_load, fz_load, mx_load, my_load, mz_load)))

            if isinstance(self.ft_sensor, BotaSerialSensor):
                if self.ft_sensor.is_functional:
                    self.use_ft_data = self.ft_sensor.start()           # start the sensor, and set flag to indicate that force-torque sensor is used
                    self.ft_sensor.calibrate()                          # calibrate the sensor

                else:
                    self.use_ft_data = False            # flag to indicate whether the force-torque sensor is used
                    rospy.logerr("The force-torque sensor is not functional. Zero interaction wrench will be used.")

            else:
                RuntimeError("The force-torque sensor object is not of the correct type")


    def createMPCfunctionWithoutInitialGuesses(self):
        """
        Instantiates the CasADi function that maps between parameters of the NLP to its optimal solutions.
        Parameters can represent initial state and bounds on the variables (that can differ at each iteration),
        while the optimal solution should be optimal state trajectory and control inputs.
        """
        self.MPC_iter = self.nlps_module.createOptimalMapWithoutInitialGuesses()

    def createMPCfunctionInitialGuesses(self):
        """
        Instantiates the CasADi function that maps between parameters of the NLP to its optimal solutions.
        Parameters can represent initial state and bounds on the variables (that can differ at each iteration),
        while the optimal solution should be optimal state trajectory and control inputs. The initial guesses for
        the primal and dual variables should also be input to the function.
        """
        self.MPC_iter_initGuess = self.nlps_module.createOptimalMapInitialGuesses()


    def setStrainMapsParamsDict(self, params_dict):
        """
        This function is used to set the dictionary of strain map parameters for later use.
        Each strain map is represented as a sum of 2D Gaussian functions (each having 6 parameters)
        """
        self.strainmaps_params_dict = params_dict

        # check if the dictionary contains various activation levels for the first value of axial rotation
        if params_dict is not None:
            if isinstance(params_dict['all_params_gaussians'][0], list):
                self.varying_activation = True
        else:
            self.varying_activation = "override"      # hack to allow us to prescribe an activation level

    
    def setActivationBounds(self, max_activation, min_activation, delta_activation):
        """
        This function allows to modify the default activation bounds considered in the trajectory optimization,
        as well as the discretization that the strain maps capture.
        """
        self.max_activation = max_activation
        self.min_activation = min_activation
        self.delta_activation = delta_activation


    def setStrainMapCurrent(self, strainmap_params):
        """"
        This function is used to manually update the strain map that is being considered, based on user input.
        It is useful just for debugging, then the update of the strain maps should be done based on the 
        state_values_current returned by sensor input (like the robotic encoder, or visual input)
        """
        self.strainmap_current = strainmap_params


    def setActivationLevel(self, activation_level):
        """
        This function allows to set the activation level that is currently considered. This call only makes sense if
        we are navigating strain maps that are also a function of the activation of the muscles.
        By setting a different activation level, the 2D strain map that is navigated will change accordingly.
        """
        assert self.varying_activation is not False, "Setting the activation level will not work, the strains do not account for it"
        if self.varying_activation!="override":
            assert self.max_activation >= activation_level, "The desired activation exceeds the bounds"
        assert self.min_activation <= activation_level, "The desired activation exceeds the bounds"
        self.activation_level = int(np.floor((activation_level - self.min_activation)/self.delta_activation))


    def upsampleSolution(self, solution, N, T, target_freq):
        """
        The function is dedicated to up-sampling a given solution for the trajectory of the shoulder pose
        (consisting of N points over a time horizon of T seconds) to a target frequency required.
        Note that only the generalized coordinates are up-sampled, as we do not care about the velocities.
        The interpolation is linear over the initial data points! Using SciPy, we could interpolate also in
        a different way.
        """
        required_points = int(np.ceil(target_freq*T))
        upsampled_indices = np.linspace(0, solution.shape[1] - 1, required_points)
        solution = solution[0::2, :]
        upsampled_solution = np.array([np.interp(upsampled_indices, np.arange(solution.shape[1]), row) for row in solution])
        return upsampled_solution
    

    def publishCartRef(self, shoulder_pose_ref, torque_ref, position_gh_in_base, base_R_sh, dist_gh_elbow, adjust_offset = False):
        """"
        This function publishes a given reference shoulder state as the equivalent 6D cartesian pose corresponding
        to the position of the elbow tip, expressed the world frame. The center of the shoulder in this frame needs 
        to be given by the user, so that they need to specify the position of the GH joint center in the world frame 
        (px, py, pz), and the orientation of the shoulder reference frame (i.e., the scapula reference frame) as well.
        The underlying assumption is that the scapula/shoulder frame remains fixed over time wrt the world frame.

        The inputs are:
            - shoulder_pose_ref: 3x1 numpy array, storing the values of plane of elevation, shoulder 
              elevation and axial rotation at a given time instant
            - torque_ref: 2x1 numpy array, storing the torques to be applied to the plane of elevation and 
              shoulder elevation (output of the trajectory optimization step)
            - position_gh_in_base: the coordinates (px, py, pz) as a numpy array
            - base_R_sh: rotation matrix defining the orientation of the shoulder frame wrt the world frame
                         (as a scipy.spatial.transform.Rotation object)
            - dist_gh_elbow: the vector expressing the distance of the elbow tip from the GH center, expressed in the 
                             shoulder frame when plane of elevation = shoulder elevation =  axial rotation= 0

        The positions and distances must be expressed in meters, and the rotations in radians.
        """
        # distinguish between the various coordinates, so that debugging is simpler
        pe = shoulder_pose_ref[0]
        se = shoulder_pose_ref[1]
        ar = shoulder_pose_ref[2]

        # define the required rotations
        base_R_elb = base_R_sh*R.from_euler('y', pe)*R.from_euler('x', -se)*R.from_euler('y', ar-ar_offset)

        base_R_ee = base_R_elb * R.from_euler('x', -np.pi/2)

        euler_angles_cmd = base_R_ee.as_euler('xyz') # store also equivalent Euler angles

        # find position for the end-effector origin
        ref_cart_point = np.matmul(base_R_elb.as_matrix(), dist_gh_elbow) + position_gh_in_base

        # modify the reference along the Z direction, to account for the increased interaction force
        # due to the human arm resting on the robot. We do this only if we are not in simulation.
        if torque_ref is not None:
            k_z = ee_stiffness[2]
            se_estimated = self.state_values_current[2]
            torque_se = torque_ref[1]
            z_current = self.current_ee_pose[2]

            new_z_ref = z_current + torque_se/(k_z * experimental_params['L_tot'] * np.sin(se_estimated))
            
            # append new z reference (in this way, we can filter both the new and the old)
            ref_cart_point = np.hstack((ref_cart_point, new_z_ref))

        # check if filter has been initialized already
        if not self.filter_initialized:
            self.last_cart_ee_cmd = ref_cart_point
            self.last_rotation_ee_cmd = euler_angles_cmd
            self.filter_initialized = True

        # when we start computing the adjusted z reference, update the state of the filter too
        if self.last_cart_ee_cmd.shape[0] < ref_cart_point.shape[0]:
            self.last_cart_ee_cmd = np.hstack((self.last_cart_ee_cmd, self.last_cart_ee_cmd[2]))
        
        # filter position to command to the robot
        ref_cart_point = self.alpha_p * ref_cart_point + (1-self.alpha_p)*self.last_cart_ee_cmd

        # substitute the old (uncorrected) z reference with the new one
        if torque_ref is not None:
            if self.simulation == 'false':
                alternative_z_ref = np.atleast_2d(ref_cart_point[2])    # save the filtered old (uncorrected) ref
                ref_cart_point = np.delete(ref_cart_point, [2])         # now ref_cart point contains [x,y, gravity_compensated_z] filtered
            else:
                alternative_z_ref = np.atleast_2d(ref_cart_point[3])    # save the gravity compensated reference (unused)
                ref_cart_point = np.delete(ref_cart_point, [3])         # now ref_cart point contains [x,y, gravity_uncompensated_z] filtered
        else:
            alternative_z_ref = np.atleast_2d(np.array([np.nan]))

        # filter orientation to command to the robot (through equivalent Euler angles)
        euler_angles_cmd = self.alpha_r * euler_angles_cmd + (1-self.alpha_r)*self.last_rotation_ee_cmd

        # update the filter state
        self.last_cart_ee_cmd = ref_cart_point
        self.last_rotation_ee_cmd = euler_angles_cmd

        # retrieve quaternion from filtered angles
        base_R_ee = R.from_euler('xyz', euler_angles_cmd)
        quaternion_cmd = base_R_ee.as_quat(scalar_first=False)

        # build the message
        message_ref = PoseStamped()
        message_ref.header.stamp = rospy.Time.now()
        message_ref.pose.position.x = ref_cart_point[0]
        message_ref.pose.position.y = ref_cart_point[1]
        message_ref.pose.position.z = ref_cart_point[2]
        message_ref.pose.orientation.x = quaternion_cmd[0]
        message_ref.pose.orientation.y = quaternion_cmd[1]
        message_ref.pose.orientation.z = quaternion_cmd[2]
        message_ref.pose.orientation.w = quaternion_cmd[3]

        # publish the message
        self.pub_trajectory.publish(message_ref)

        # publish also the alternative_z_ref
        message_z = Float32MultiArray()
        message_z.layout.data_offset = 0
        message_z.data = alternative_z_ref
        self.pub_z_level.publish(message_z)
        

    def publishInitialPoseAsCartRef(self, shoulder_pose_ref, position_gh_in_base, base_R_sh, dist_gh_elbow):
        """"
        This function publishes a given reference shoulder state as the equivalent 6D cartesian pose corresponding
        to the position of the elbow tip, expressed the world frame. The center of the shoulder in this frame needs 
        to be given by the user, so that they need to specify the position of the GH joint center in the world frame 
        (px, py, pz), and the orientation of the shoulder reference frame (i.e., the scapula reference frame) as well.
        The underlying assumption is that the scapula/shoulder frame remains fixed over time wrt the world frame.

        The inputs are:
            - shoulder_pose_ref: 3x1 numpy array, storing the values of plane of elevation, shoulder 
              elevation and axial rotation at a given time instant
            - position_gh_in_base: the coordinates (px, py, pz) as a numpy array
            - base_R_sh: rotation matrix defining the orientation of the shoulder frame wrt the world frame
                         (as a scipy.spatial.transform.Rotation object)
            - dist_gh_elbow: the vector expressing the distance of the elbow tip from the GH center, expressed in the 
                             shoulder frame when plane of elevation = shoulder elevation =  axial rotation= 0

        The positions and distances must be expressed in meters, and the rotations in radians.

        This function is conceptually equivalent to publishCartRef, but it uses another ROS publisher to publish the
        required reference to the topic associated to the initial pose for the robot. It will keep publishing the initial pose
        until the robot controller receives it, and then it stops.

        Once called, it will also return the position that the robot has been commanded to reach.
        """
        shoulder_state = np.zeros((6, 1))
        shoulder_state[0::2] = shoulder_pose_ref.reshape(-1, 1)
        self.x_opt = shoulder_state

        # perform extra things if this is the first time we execute this
        if not self.flag_pub_trajectory:
            # We need to set up the structure to deal with the new thread, to allow 
            # continuous publication of the optimal trajectory
            self.publish_thread = threading.Thread(target=self.publish_continuous_trajectory, 
                                                    args = (position_gh_in_base, base_R_sh, dist_gh_elbow))   # creating the thread
            
            self.publish_thread.daemon = True   # this allows to terminate the thread when the main program ends
            self.flag_pub_trajectory = True     # update flag
            self.publish_thread.start()         # start the publishing thread

            # start the thread dedicated to the estimation of the muscle activation as well
            self.rmr_thread = threading.Thread(target=self.estimate_muscle_activation, args=())   # creating the thread
            self.rmr_thread.daemon = True   # this allows to terminate the thread when the main program ends
            self.rmr_thread.start()


    def _shoulder_pose_cb(self, data):
        """
        Callback receiving and processing the current shoulder pose.
        """
        if not self.flag_receiving_shoulder_pose:       # if this is the first time we receive something, update flag
            self.flag_receiving_shoulder_pose = True

        # retrieve the current state as estimated on the robot's side
        self.state_values_current = np.array(data.data[0:6])        # update current pose
        self.state_dot_current = np.array([data.data[i] for i in [1, 6, 3, 7, 5, 8]])   # update current derivative of the pose

        if not self.speed_estimate:                                 # choose whether we use the velocity estimate or not
            self.state_values_current[1::2] = 0

        time_now = time.time()

        if experiment == 3 and self.time_begin_optimizing is not None:
            self.state_values_current[4] = np.arcsin(np.sin((time_now-self.time_begin_optimizing)/6))/2

        self.current_ee_pose = np.array(data.data[-3:])             # update estimate of cartesian position of the robot EE
        self.current_ee_orientation = np.array(data.data[-7:-3])    # update estimate of orientation of the robot EE (as a quaternion, scalar last)

        # after receiving the values, check on which strain map we are moving
        # we need to approximate the value of the axial rotation for this.
        self.strainmap_current = np.argmin(np.abs(self.nlps_module.ar_values - np.rad2deg(self.state_values_current[4])))   # index of the strain map

        rounded_ar = self.nlps_module.ar_values[self.strainmap_current] # rounded value of the axial rotation

        # set the strain map parameters in the NLP, based on the current state
        if self.varying_activation is False:
            self.nlps_module.setParametersStrainMap(self.strainmaps_params_dict['num_gaussians'][self.strainmap_current], 
                                                    self.strainmaps_params_dict['all_params_gaussians'][self.strainmap_current],
                                                    rounded_ar)
        elif self.varying_activation is True:
            self.nlps_module.setParametersStrainMap(self.strainmaps_params_dict['num_gaussians'][self.strainmap_current][self.activation_level], 
                                                    self.strainmaps_params_dict['all_params_gaussians'][self.strainmap_current][self.activation_level],
                                                    rounded_ar,
                                                    self.activation_level*self.delta_activation + self.min_activation)
            
        elif self.varying_activation=="override":
            # here we hardcode some particular activation levels, to design meaningful strain maps for the experiments
            if self.activation_level == 180:
                p1 = np.array([0, 0, 0, 1, 1, 0])
                p2 = np.array([0, 0, 0, 1, 1, 0])
                p3 = np.array([0, 0, 0, 1, 1, 0])
                params_gaussians = np.hstack((p1, p2, p3))
                self.nlps_module.setParametersStrainMap(3, params_gaussians, rounded_ar, None)

            elif self.activation_level == 182:
                p1 = np.array([3, 72/160, 80/144, 10/160, 5/144, 0])
                p2 = np.array([0, 0, 0, 1, 1, 0])
                p3 = np.array([0, 0, 0, 1, 1, 0])
                params_gaussians = np.hstack((p1, p2, p3))
                self.nlps_module.setParametersStrainMap(3, params_gaussians, rounded_ar, None)

            elif self.activation_level == 184:
                p1 = np.array([3, 33/160, 67/144, 10/160, 5/144, 0])
                p2 = np.array([0, 0, 0, 1, 1, 0])
                p3 = np.array([0, 0, 0, 1, 1, 0])
                params_gaussians = np.hstack((p1, p2, p3))
                self.nlps_module.setParametersStrainMap(3, params_gaussians, rounded_ar, None)

            elif self.activation_level >190:
                p1 = np.array([2.5, 0.45, 0.5, 0.075, 0.15, 0])
                p2 = np.array([0, 0, 0, 1, 1, 0])
                p3 = np.array([0, 0, 0, 1, 1, 0])
                params_gaussians = np.hstack((p1, p2, p3))

                self.nlps_module.setParametersStrainMap(3, params_gaussians, rounded_ar, None)

            else:
                # extra case to allow for some flexibility in testing
                p1 = np.array([0, 0, 0, 1, 1, 0])
                p2 = np.array([0, 0, 0, 1, 1, 0])
                p3 = np.array([0, 0, 0, 1, 1, 0])
                params_gaussians = np.hstack((p1, p2, p3))
                self.nlps_module.setParametersStrainMap(3, params_gaussians, rounded_ar, None)

        # we also need to update the load effect of the sensor, based on the current EE pose
        weight_load_in_world = self.gravity_in_world*self.sensor_load_mass    # compute the gravity force in the world frame

        # express load into sensor frame
        sensor_R_world = self.sensor_R_ee * R.from_quat(self.current_ee_orientation, scalar_first=False).inv()
        weight_in_sensor = np.matmul(sensor_R_world.as_matrix(), weight_load_in_world)
        mx_load = weight_in_sensor[1]*self.sensor_load_com[2] - weight_in_sensor[2]*self.sensor_load_com[1]
        my_load = weight_in_sensor[2]*self.sensor_load_com[0] - weight_in_sensor[0]*self.sensor_load_com[2]
        mz_load = weight_in_sensor[0]*self.sensor_load_com[1] - weight_in_sensor[1]*self.sensor_load_com[0]
        self.ft_sensor.set_load_effect(np.array((weight_in_sensor[0], weight_in_sensor[1], weight_in_sensor[2], mx_load, my_load, mz_load)))
        
        # set up a logic that allows to start and update the visualization of the current strain map at a given frequency
        # We set this fixed frequency to be 10 Hz
        if np.round(time_now-np.fix(time_now), 2) == np.round(time_now-np.fix(time_now),1):
            self.nlps_module.strain_visualizer.updateStrainMap(self.nlps_module.all_params_gaussians, 
                                                                self.state_values_current[[0,2]], 
                                                                self.x_opt[[0,2],:],
                                                                self.nlps_module.goal[[0,2]], 
                                                                self.state_values_current[[1,3]])


    def _flag_run_optimization_cb(self, data):
        """
        This callback catches the boolean message that is published on a default topic, informing whether the robot wants
        to receive new, optimized references or not.
        """
        self.flag_run_optimization = data.data
        self.flag_pub_trajectory = data.data

    
    def keepOptimizing(self):
        """
        This function retrieves the value stored in self.flag_run_optimization, allowing to continue or stop the trajectory
        optimization computations.
        """
        return self.flag_run_optimization


    def waitForShoulderState(self):
        """"
        Utility that stops the execution of the code until the current shoulder pose becomes available.
        This ensures that correct communication is established between the biomechanics simulation and the robot.
        """
        while not rospy.is_shutdown() and not self.flag_receiving_shoulder_pose:
            self.ros_rate.sleep()

        print("Receiving current shoulder pose.")

    
    def optimize_trajectory(self, x_goal, delay):
        """
        This function computes the optimal trajectory towards the goal position, given the current shoulder state
        and the estimated delay at which the solution will become available. This last input is an estimate of how
        long the optimization will take, so that the initial state of the system is changed accordingly, assuming that
        the current movement will continue as-is.
        The optimal trajectory and set of controls to follow it are saved, to be executed by the robot.
        NOTE: The optimization is not warm-started.
        """
        # initialize flag to monitor if the solve found an optimal solution
        failed = 0

        # retrieve the initial state of the system, plus an estimated displacement given the current velocities
        estimated_displacement = delay * np.array([self.state_values_current[1], 
                                                   0, 
                                                   self.state_values_current[3], 
                                                   0,
                                                   self.state_values_current[5],
                                                   0])

        initial_state = self.state_values_current + estimated_displacement                  # estimated shoulder pose

        # if we are considering strain in our formulation, add the parameters of the strain map to the numerical input
        if self.nlps_module.num_gaussians>0:
            params_g1 = self.nlps_module.all_params_gaussians[0:6]
            params_g2 = self.nlps_module.all_params_gaussians[6:12]
            params_g3 = self.nlps_module.all_params_gaussians[12:18]

            # solve the NLP problem given the current state of the system (with strain information)
            # we solve the NLP, and catch if there was an error. If so, notify the user and retry
            try:
                time_start = time.time()
                u_opt, x_opt, j_opt, _, strain_opt, xddot_opt = self.MPC_iter(initial_state, x_goal, self.state_values_current[4], params_g1, params_g2, params_g3)
                time_execution = time.time()-time_start
                strain_opt = strain_opt.full().reshape(1, self.nlps_module.N+1, order='F')

            except Exception as e:
                print('Solver failed:', e)
                print('retrying ...')
                failed = 1

        else:
            # solve the NLP problem given the current state of the system (without strain information)
            # we solve the NLP, and catch if there was an error. If so, notify the user and retry
            try:
                time_start = time.time()
                u_opt, x_opt, _,  j_opt, xddot_opt = self.MPC_iter(initial_state, x_goal, self.state_values_current[4])
                time_execution = time.time() - time_start

                strain_opt = np.nan * np.ones(np.shape(x_opt))
            except Exception as e:
                print('Solver failed:', e)
                print('retrying ...')
                failed = 1

        self.nlp_count += 1         # update the number of iterations until now
        self.failed_count += failed # update number of failed iterations

        # only do the remaining steps if we have a new solution
        if not failed:
            # convert the solution to numpy arrays, and store them to be processed
            x_opt = x_opt.full().reshape(self.nlps_module.dim_x, self.nlps_module.N+1, order='F')
            u_opt = u_opt.full().reshape(self.nlps_module.dim_u, self.nlps_module.N, order='F')
            xddot_opt = xddot_opt.full().reshape(int(self.nlps_module.dim_x/2), self.nlps_module.N, order='F')

            # save the strain value
            self.strain_opt = strain_opt
            
            # update the optimal values that are stored in the TO module. 
            # They can be accessed only if there is no other process that is modifying them
            with self.x_opt_lock:
                # self.x_opt = x_opt[:, 1::]      # the first point is discarded, as it is the current one
                self.x_opt = x_opt[:, 2::]      # the first points are discarded, to compensate for relatively low stiffness of the controller
                self.u_opt = u_opt[:,1::]       # same as above, to guarantee consistency

            # update average running time
            self.avg_nlp_time = ((self.avg_nlp_time * self.nlp_count) + time_execution) / (self.nlp_count+1)

            # update the optimal values that are stored in the NLPS module as well
            self.nlps_module.x_opt = x_opt      
            self.nlps_module.u_opt = u_opt

            # publish the optimal values to a topic, so that they can be recorded during experiments
            message = Float32MultiArray()
            u_opt = np.concatenate((u_opt, np.atleast_2d(np.nan*np.ones((2,1)))), axis = 1)  # adding one NaN to match dimensions of other arrays
            activation = self.activation_level*self.delta_activation + self.min_activation
            message.data = np.hstack((np.vstack((x_opt, u_opt, strain_opt)).flatten(), activation))    # stack the three outputs in a single message (plus activation), and flatten it for publishing
            self.pub_optimization_output.publish(message)

            return u_opt, x_opt, j_opt, strain_opt, xddot_opt
        

    def optimize_trajectory_astar(self, maze, x_goal, strainmap_list):
        """
        This function plans the trajectory towards the goal position, given the current shoulder state.
        It uses a modified version of the A* algorithm, as presented in 
        "Biomechanics aware collaborative robot system for delivery of safe physical therapy in shoulder rehabilitation", 
        from Prendergast et al. (RAL 2021).
        The path to follow on the current strain map is saved, to be executed by the robot.
        """
        # initialize flag to monitor if the solve found a solution
        failed = 0

        init_pose = tuple(np.round(self.state_values_current[[0,2]]).astype(int))   # initial pose of the shoulder(pe, se)
        end_pose = tuple(np.round(x_goal[[0,2]]).astype(int))                       # goal pose of the shoulder(pe, se)

        fullpath = self.astar_planner.plan(maze, init_pose, end_pose, strainmap_list)

        pathpnts = ([t[0] for t in fullpath], [t[1] for t in fullpath])
        pe = pathpnts[0]
        se = pathpnts[1]
        ar = np.ones(np.shape(pe))*self.state_values_current[4]  # axial rotation is constant
        
        planned_states = np.zeros((self.nlps_module.dim_x, len(pe)))
        planned_states[0,:] = pe
        planned_states[2,:] = se
        planned_states[4,:] = ar

        planned_controls = np.zeros((self.nlps_module.dim_u, len(pe)-1))    # A* cannot return the controls!

        # update the optimal values stored
        with self.x_opt_lock:
            self.x_opt = planned_states
            self.u_opt = planned_controls



    def publish_continuous_trajectory(self, p_gh_in_base, rot_ee_in_base_0, dist_shoulder_ee):
        """
        This function picks the most recent information regarding the optimal shoulder trajectory,
        converts it to end effector space and publishes the robot reference continuously. A flag enables/disables
        the computations/publishing to be performed, such that this happens only if the robot controller needs it.
        """
        rate = rospy.Rate(1/self.nlps_module.h)  # Set the publishing rate (depending on the parameters of the NLP)

        while not rospy.is_shutdown():
            if self.flag_pub_trajectory:    # perform the computations only if needed
                with self.x_opt_lock:       # get the lock to read and modify references
                    if self.x_opt is not None:
                        # We will pick always the first element of the trajectory, and then delete it (unless it is the last element)
                        # This is done in blocking mode to avoid conflicts, then we move on to processing the shoulder pose and convert
                        # it to cartesian trajectory
                        if np.shape(self.x_opt)[1]>1:
                            # If there are many elements left, get the first and then delete it
                            cmd_shoulder_pose = self.x_opt[0::2, 0]
                            self.x_opt = np.delete(self.x_opt, obj=0, axis=1)   # delete the first column

                            if self.u_opt is not None:
                                cmd_torques = self.u_opt[:,0]
                                self.u_opt = np.delete(self.u_opt, obj=0, axis=1)
                            else:
                                cmd_torques = None

                        else:
                            # If there is only one element left, keep picking it
                            cmd_shoulder_pose = self.x_opt[0::2, 0]
                            if self.u_opt is not None:
                                cmd_torques = self.u_opt[:,0]
                            else:
                                cmd_torques = None

                        self.publishCartRef(cmd_shoulder_pose, cmd_torques, p_gh_in_base, rot_ee_in_base_0, dist_shoulder_ee, adjust_offset = False)

            rate.sleep()


    def estimate_muscle_activation(self):
        """
        This function estimates the current activation level of the human shoulder muscles, based on:
        - a personalized biomechanical model
        - the current position, velocity and accelerations of the subject
        - the interaction wrenches measured by the force-torque sensor
        """
        while not rospy.is_shutdown():
            if self.flag_receiving_shoulder_pose:
                # retrieve position, velocity and acceleration of the shoulder DoFs (order is always pe, se, ar)
                position_sh = self.state_values_current[0::2]
                velocity_sh = self.state_values_current[1::2]
                acceleration_sh = self.state_dot_current[1::2]
                
                # # retrieve the interaction wrenches from the force-torque sensor
                # if self.ft_sensor.is_functional:
                #     # get the interaction wrench from the force-torque sensor
                #     interaction_wrench = np.array([self.ft_sensor.current_reading.fx,
                #                                 self.ft_sensor.current_reading.fy,
                #                                 self.ft_sensor.current_reading.fz,
                #                                 self.ft_sensor.current_reading.mx,
                #                                 self.ft_sensor.current_reading.my,
                #                                 self.ft_sensor.current_reading.mz])
                # else:
                #     interaction_wrench = np.zeros(6)
                # # TODO: this is still work in progress
                # # estimate the muscle activation level for all the muscles in the model
                # current_activation, _, info = self.rmr_solver.solve(time.time(), position_sh, velocity_sh, acceleration_sh, interaction_wrench)

                # # publish the activation level (for debugging and logging)
                # message = Float32MultiArray()
                # message.data = current_activation
                # self.pub_activation.publish(message)

                # # publish the interaction wrenches after gravity compensation (for debugging and logging)
                # message = Float32MultiArray()
                # message.data = interaction_wrench
                # self.pub_compensated_wrench.publish(message)

            # sleep for a while
            self.ros_rate.sleep()


    def setOptimalReferenceToCurrentPose(self):
        """
        This function allows to overwrite the optimal state/control values. They are
        substituted by the current state of the subject/patient sensed by the robot.
        """
        with self.x_opt_lock:
            self.x_opt = self.state_values_current.reshape(self.nlps_module.dim_x, 1)
            self.u_opt = np.zeros((2,1))


    def setOptimalReferenceToCurrentOptimalPose(self):
        """
        This function allows to overwrite the optimal state/control values, by keeping
        only the one that is currently being tracked.
        """
        with self.x_opt_lock:
            self.x_opt = self.x_opt[:,0].reshape((6,1))
            self.u_opt = np.zeros((2,1))


    def reachedPose(self, pose, tolerance = 0):
        """
        This function checks whether the current position of the human model
        is the same as the one we are willing to reach ('pose'). Optionally, 
        a 'tolerance' on such pose can be considered too, in radians.
        """
        if np.linalg.norm(self.state_values_current[[0,2]] - pose[[0,2]]) <= tolerance:
            return True
        else:
            return False


    def getStats(self, flag_print=Bool):
        """
        This utility allows to get some statistics at the end of an execution.
        """
        stats = {}

        stats['avg_time_opt'] = self.avg_nlp_time
        stats['num_opt'] = self.nlp_count
        stats['num_failed_opt'] = self.failed_count

        if flag_print:
            print("\n--------------------------------------------------------")
            print("Average optimization time: ", np.round(self.avg_nlp_time,3))
            print("# optimizations: ", self.nlp_count)
            print("# failed optimizations: ", self.failed_count)
            print("--------------------------------------------------------\n")

        return stats
    
    def setSensorLoadParameters(self, mass, com):
        """
        This function allows to set the mass and center of mass of the load that is being handled by the force-torque
        sensor. The mass is expressed in kg, while the center of mass is expressed in meters.
        """
        self.sensor_load_mass = mass
        self.sensor_load_com = com


class nlps_module():
    """
    Class defining the nonlinear programming (NLP) problem to be solved at each iteration of the 
    trajectory optimization (TO) algorithm. It leverages CasADi and OpenSim to find the optimal 
    trajectory that the OpenSim model should follow in order to navigate the corresponding 
    strain-space (defined by a given strain parameters, aka "strain map"). 

    We call this a nlps_module (where the "s" stands for "strain")
    """
    def __init__(self):
        """"
        Initialization in which we set only empty variables that will be modified by the user
        through dedicated functions.
        """
        self.T = None                   # time horizon for the optimization
        self.N = None                   # number of control intervals
        self.h = None                   # duration of each control interval

        # Initial condition and final desired goal (both expressed in state-space)
        self.x_0 = None
        self.goal = None

        # CasADi symbolic variables
        self.x = None                   # CasADi variable indicating the state of the system (position and velocities of relevant coordinates)
                                        # In our case, the coordinates we care about are plane of elevation (pe) and shoulder elevation (se)
        
        self.u = None                   # CasADi variable representing the control vector to be applied to the system
                                        # In our case, they are forces to be applied to the human at their elbow

        self.x_d = None                 # CasADi variable representing the desired final position on the strai nmap

        # Naming and ordering of states and controls
        self.state_names = None
        self.xdim = 0
        self.control_names = None
        self.dim_u = 0

        # Systems dynamics (as a CasADi function, or coming from the OpenSim model as a CasADi Callback)
        self.sys_dynamics = None

        # strain map parameters
        self.num_gaussians = 0          # the number of 2D Gaussians used to approximate the current strain map
        self.num_params_gaussian = 6    # number of parameters that each Gaussian is defined of (for a 2D Gaussian, we need 6)
        self.all_params_gaussians = []  # list containing the values of all the parameters defining all the Gaussians
                                        # For each Gaussian, we have: amplitude, x0, y0, sigma_x, sigma_y, offset
                                        # (if more Gaussians are present, the parameters of all of them are concatenated)
        
        self.pe_boundaries = [-20, 160] # defines the interval of physiologically plausible values for the plane of elevation [deg]
        self.se_boundaries = [0, 144]   # as above, for the shoulder elevation [deg]
        self.ar_boundaries = [-90, 100] # as above, for the axial rotation [deg]

        self.strainmap_step = 4         # discretization step used along the model's coordinate [in degrees]
                                        # By default we set it to 4, as the strain maps are generated from the biomechanical model
                                        # with this grid accuracy
        
        self.ar_values = np.arange(self.ar_boundaries[0], self.ar_boundaries[1], self.strainmap_step)
        
        # strain map parameters (for visualization)
        self.pe_datapoints = np.array(np.arange(self.pe_boundaries[0], self.pe_boundaries[1], self.strainmap_step))
        self.se_datapoints = np.array(np.arange(self.se_boundaries[0], self.se_boundaries[1], self.strainmap_step))

        self.X,self.Y = np.meshgrid(self.pe_datapoints, self.se_datapoints, indexing='ij')
        self.X_norm = self.X/np.max(np.abs(self.pe_boundaries))
        self.Y_norm = self.Y/np.max(np.abs(self.se_boundaries))

        self.strain_visualizer = RealTimeStrainMapVisualizer(self.X_norm, self.Y_norm, self.num_params_gaussian, self.pe_boundaries, self.se_boundaries)

        # Type of collocation used and corresponding matrices
        self.pol_order = None
        self.collocation_type = None
        self.B = None                   # quadrature matrix
        self.C = None                   # collocation matrix
        self.D = None                   # end of the intervaprobleml

        # cost function
        self.cost_function = None       # CasADi function expressing the cost to be minimized
        self.gamma_goal = 0             # weight of the distance to the goal [1/(rad^2)]
        self.gamma_strain = 0           # weight of the strain value in the cost function
        self.gamma_velocities = 0       # weight for the coordinates' velocities
        self.gamma_acceleration = 0     # weight for the coordinates' accelerations

        # constraints
        self.constrain_final_pos = False    # constrain the final position wrt the goal
        self.eps_final_pos = 0              # tolerance for the position
        self.constrain_final_vel = False    # constrain the final velocity wrt the goal
        self.eps_final_vel = 0              # tolerance for velocity

        # CasADi optimization problem
        self.opti = ca.Opti()
        self.nlp_is_formulated = False              # has the nlp being built with formulateNLP()?
        self.solver_options_set = False             # have the solver options being set by the user?
        self.default_solver = 'ipopt'
        self.default_solver_opts = {'ipopt.print_level': 0,
                                    'print_time': 0, 
                                    'error_on_fail': 1,
                                    'ipopt.tol': 1e-3,
                                    'expand': 0,
                                    'ipopt.hessian_approximation': 'limited-memory'}


        # parameters of the NLP
        self.params_list = []           # it collects all the parameters that are used in a single instance
                                        # of the NLP
        
        # solution of the NLP
        self.Xs = None                  # the symbolic optimal trajectory for the state x
        self.Xddot_s = None             # the symbolic expression for the optimal accelerations of the model's DoFs
        self.Us = None                  # the symbolic optimal sequence of the controls u
        self.Js = None                  # the symbolic expression for the cost function
        self.strain_s = None            # the symbolic strain history, as a function of the optimal solution
        self.solution = None            # the numerical solution (where state and controls are mixed together)
        self.x_opt = None               # the numerical optimal trajectory for the state x
        self.u_opt = None               # the numerical optimal sequence of the controls u
        self.strain_opt = None          # the numerical values of the strain, along the optimal trajectory
        self.lam_g0 = None              # initial guess for the dual variables (populated running solveNLPonce())
        self.solution_is_updated = False


    def setTimeHorizonAndDiscretization(self, N, T):
        self.T = T      # time horizon for the optimal control
        self.N = N      # numeber of control intervals
        self.h = T/N

    def setSystemDynamics(self, sys_dynamics):
        """
        Utility to set the system dynamics. It can receive as an input either a CasADi callback
        (if an OpenSim model is to be used) or a CasADi function directly (if the ODE are know analytically).
        """
        self.sys_dynamics = sys_dynamics
    

    def initializeStateVariables(self, x, names):
        self.x = x
        self.state_names = names
        self.dim_x = x.shape[0]


    def initializeControlVariables(self, u, names):
        self.u = u
        self.control_names = names
        self.dim_u = u.shape[0]


    def populateCollocationMatrices(self, order_polynomials, collocation_type):
        """
        Here, given an order of the collocation polynomials, the corresponding matrices are generated
        The collocation polynomials used are Lagrange polynomials.
        The collocation type determines the collocation points that are used ('legendre', 'radau').
        """
        # Degree of interpolating polynomial
        self.pol_order = order_polynomials

        # Get collocation points
        tau = ca.collocation_points(self.pol_order, collocation_type)

        # Get linear maps
        self.C, self.D, self.B = ca.collocation_coeff(tau)

    
    def setCostFunction(self, cost_function):
        """"
        The user-provided CasADi function is used as the cost function of the problem
        """
        self.cost_function = cost_function


    def enforceFinalConstraints(self, on_position, on_velocity, eps_pos = 0, eps_vel = 0.01):
        """
        This function allows to choose whether to enforce constraints on the final position and
        velocity in the NLP. The user can input the tolerances with which the constraints will be respected
        (in rad and rad/s respectively)
        """
        self.constrain_final_pos = on_position
        self.eps_final_pos = eps_pos
        self.constrain_final_vel = on_velocity
        self.eps_final_vel = eps_vel


    def setInitialState(self, x_0):
        self.x_0 = x_0


    def setGoal(self, goal):
        self.goal = goal
     
    
    def formulateNLP_callbackDynamics(self, constraint_list, initial_guess_prim_vars = None, initial_guess_dual_vars = None):
        """"
        This function creates the symbolic structure of the NLP problem.
        For now it is still not very general, needs to be rewritten if different problems
        are to be solved.
        """

        if self.goal is None:
            RuntimeError("Unable to continue. The goal of the NLP has not been set yet. \
                         Do so with setGoal()!")
        if self.x_0 is None:
            RuntimeError("Unable to continue. The initial state of the NLP has not been set yet. \
                         Do so with setInitialState()!")
            
        if self.sys_dynamics is None:
            RuntimeError("Unable to continue. The system dynamics have not been specified. \
                         Do so with setSystemDynamics()!")
            
        if self.cost_function is None:
            RuntimeError("Unable to continue. The cost function have not been specified. \
                         Do so with setCostFunction()!")
            
        if len(self.all_params_gaussians) == self.num_gaussians * 6:
            if len(self.all_params_gaussians)==0:
                print("No (complete) information about strain maps have been included \nThe NLP will not consider them")
        else:
            RuntimeError("Unable to continue. The specified strain maps are not correct. \
                         Check if the parameters provided are complete wrt the number of Gaussians specified. \
                         Note: we assume that 6 parameters per (2D) Gaussian are given")
            
        J = 0

        u_max = constraint_list['u_max']
        u_min = constraint_list['u_min']

        # initialize empty list for the parameters used in the problem
        # the parameters collected here can be changed at runtime   
        self.params_list = []

        # if this information is present, define the strain map to navigate onto
        if self.num_gaussians>0:
            # for now, let's assume that there will always be 3 Gaussians (if this is not true, consider
            # if it is better to have a fixed higher number or a variable one)
            tmp_param_list = []     # this is an auxiliary list, to collect all the strain-related params and 
                                    # append them at the end.
            
            # parameters of the 1st Gaussian
            p_g1 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g1)

            # definition of the 1st Gaussian (note that the state variables are normalized!)
            g1 = p_g1[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g1[1])**2/(2*p_g1[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g1[2])**2/(2*p_g1[4]**2))) + p_g1[5]

            # parameters of the 2nd Gaussian
            p_g2 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g2)

            # definition of the 2nd Gaussian (note that the state variables are normalized!)
            g2 = p_g2[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g2[1])**2/(2*p_g2[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g2[2])**2/(2*p_g2[4]**2))) + p_g2[5]

            # parameters of the 3rd Gaussian
            p_g3 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g3)

            # definition of the 3rd Gaussian (note that the state variables are normalized!)
            g3 = p_g3[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g3[1])**2/(2*p_g3[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g3[2])**2/(2*p_g3[4]**2))) + p_g3[5]

            # definition of the symbolic cumulative strain map
            strainmap = g1 + g2 + g3
            # strainmap_sym = ca.Function('strainmap_sym', [self.x], [strainmap], {"allow_free":True})
            strainmap_sym = ca.Function('strainmap_sym', [self.x, p_g1, p_g2, p_g3], [strainmap])

            # save the symbolic strain map for debugging
            self.strainmap_sym = strainmap_sym

        #  "Lift" initial conditions
        Xk = self.opti.variable(self.dim_x)
        init_state = self.opti.parameter(self.dim_x)     # parametrize initial condition
        self.params_list.append(init_state)              # add the parameter to the list of parameters for the NLP
        self.opti.subject_to(Xk==init_state)

        # parametrize goal position, and state describing axial rotation (phi and phi_dot)
        desired_state = self.opti.parameter(self.dim_x)  # parametrize desired goal
        self.params_list.append(desired_state)           # add the parameter to the list of parameters for the NLP

        phi_prm = self.opti.parameter(1)
        self.params_list.append(phi_prm)
        phi_dot_prm = self.opti.parameter(1)             # these are just internal parameters, not modifiable from outside

        # Collect all states/controls, and strain along the trajectory
        Xs = [Xk]
        Us = []
        Xddot_s = []
        strain_s = []

        # formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = self.opti.variable(self.dim_u)
            Us.append(Uk)
            self.opti.subject_to(u_min <= Uk)
            self.opti.subject_to(Uk <= u_max)

            # optimization variable (state) at collocation points
            Xc = self.opti.variable(self.dim_x, self.pol_order)

            # evaluate ODE right-hand-side at collocation points (NOTE: this now depends from the number of coordinates!)
            ode_tuple = self.sys_dynamics(Xc[0,:],      # theta at collocation points
                                          Xc[1,:],      # theta_dot " " "
                                          Xc[2,:],      # psi " " "
                                          Xc[3,:],      # psi_dot   " " "
                                          phi_prm,      # phi   (provided as parameter)
                                          phi_dot_prm,  # phi_dot (provided as parameter)
                                          Uk[0,:],  # tau_theta " " "
                                          Uk[1,:],) # tau_psi   " " "
            
            ode_MX = ca.vertcat(ode_tuple[0], ode_tuple[1], ode_tuple[2], ode_tuple[3], ode_tuple[4], ode_tuple[5])

            quad = self.cost_function(Xc, Uk, init_state, desired_state)

            # add contribution to quadrature function
            J = J + self.h*ca.mtimes(quad, self.B)

            # add acceleration values to the cost
            J = J + self.gamma_acceleration * (self.h*(ca.mtimes(ode_MX[1, :]**2, self.B))+self.h*(ca.mtimes(ode_MX[3,:]**2, self.B)))

            # add velocity values to the cost
            J = J + self.gamma_velocities * (self.h*(ca.mtimes(ode_MX[0, :]**2, self.B))+self.h*(ca.mtimes(ode_MX[2,:]**2, self.B)))
            
            # check if we care about the strain
            if self.num_gaussians>0:
                # if so, add term related to current strain to the cost
                # the strain is evaluated only at the knots of the optimization mesh
                # (note that the strain map is defined in degrees, so we convert our state to that)
                J = J + self.gamma_strain * strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3)

                # record the strain level in the current state value
                # (note that the strain map is defined in degrees, so we convert our state to that)
                strain_s.append(strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3))

            # get interpolating points of collocation polynomial
            Z = ca.horzcat(Xk, Xc)

            # get slope of interpolating polynomial (normalized)
            Pidot = ca.mtimes(Z, self.C)

            # match with ODE right-hand-side
            self.opti.subject_to(Pidot[0:4,:]==self.h*ode_MX[0:4,:])    # theta, theta_dot, psi, psi_dot (no phi as that is not tracked)
            
            # save coordinates' accelerations (only for the first collocation point)
            Xddot_s.append(ode_MX[1::2, 0])

            # state at the end of collocation interval
            Xk_end = ca.mtimes(Z, self.D)

            # new decision variable for state at the end of interval
            Xk = self.opti.variable(self.dim_x)
            Xs.append(Xk)

            # continuity constraint
            self.opti.subject_to(Xk_end==Xk)

        if self.num_gaussians>0:
            # record the strain level at the final step
            # (note that the strain map is defined in degrees, so we convert our state to that)
            strain_s.append(strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3))
            self.strain_s = ca.vertcat(*strain_s)

        # adding constraint to reach the final desired state
        if self.constrain_final_vel:
            self.opti.subject_to((Xk[1]-desired_state[1])**2<self.eps_final_vel)   # bound on final (zero) velocity
            self.opti.subject_to((Xk[3]-desired_state[3])**2<self.eps_final_vel)   # bound on final (zero) velocity
        if self.constrain_final_pos:
            self.opti.subject_to((Xk[0]-desired_state[0])**2<self.eps_final_pos)   # bound on final (zero) velocity
            self.opti.subject_to((Xk[2]-desired_state[2])**2<self.eps_final_pos)   # bound on final (zero) velocity

        self.Us = ca.vertcat(*Us)
        self.Xs = ca.vertcat(*Xs)
        self.Xddot_s = ca.vertcat(*Xddot_s)

        # explicitly provide an initial guess for the primal variables
        if initial_guess_prim_vars is not None:
            self.opti.set_initial(initial_guess_prim_vars)

        # explicitly provide an initial guess for the dual variables
        if initial_guess_dual_vars is None:
            initial_guess_dual_vars = np.zeros((self.opti.lam_g.shape))

        self.opti.set_initial(self.opti.lam_g, initial_guess_dual_vars)

        # set the values of the parameters
        self.opti.set_value(init_state, self.x_0)       # this can be changed at runtime
        self.opti.set_value(desired_state, self.goal)   # this can be changed at runtime
        self.opti.set_value(phi_prm, 0)
        self.opti.set_value(phi_dot_prm, 0)

        if self.num_gaussians>0:
            self.params_list.extend(tmp_param_list)     # append at the end the parameters for the strains

            self.opti.set_value(p_g1, self.all_params_gaussians[0:6])   # NOTE: hardcoded!
            self.opti.set_value(p_g2, self.all_params_gaussians[6:12])
            self.opti.set_value(p_g3, self.all_params_gaussians[12:18])

        # define the cost function to be minimized, and store its symbolic expression
        self.opti.minimize(J + ca.sumsqr(ca.vertcat(*self.params_list)))
        self.Js = J

        # set flag indicating process was successful
        self.nlp_is_formulated = True


    def formulateNLP_functionDynamics(self, constraint_list, initial_guess_prim_vars = None, initial_guess_dual_vars = None):
        """"
        This function creates the symbolic structure of the NLP problem.
        For now it is still not very general, needs to be rewritten if different problems
        are to be solved.
        """

        if self.goal is None:
            RuntimeError("Unable to continue. The goal of the NLP has not been set yet. \
                         Do so with setGoal()!")
        if self.x_0 is None:
            RuntimeError("Unable to continue. The initial state of the NLP has not been set yet. \
                         Do so with setInitialState()!")
            
        if self.sys_dynamics is None:
            RuntimeError("Unable to continue. The system dynamics have not been specified. \
                         Do so with setSystemDynamics()!")
            
        if self.cost_function is None:
            RuntimeError("Unable to continue. The cost function have not been specified. \
                         Do so with setCostFunction()!")
            
        if len(self.all_params_gaussians) == self.num_gaussians * 6:
            if len(self.all_params_gaussians)==0:
                print("No (complete) information about strain maps have been included \nThe NLP will not consider them")
        else:
            RuntimeError("Unable to continue. The specified strain maps are not correct. \
                         Check if the parameters provided are complete wrt the number of Gaussians specified. \
                         Note: we assume that 6 parameters per (2D) Gaussian are given")
            
        J = 0

        u_max = constraint_list['u_max']
        u_min = constraint_list['u_min']

        # initialize empty list for the parameters used in the problem
        # the parameters collected here can be changed at runtime   
        self.params_list = []

        # if this information is present, define the strain map to navigate onto
        if self.num_gaussians>0:
            # for now, let's assume that there will always be 3 Gaussians (if this is not true, consider
            # if it is better to have a fixed higher number or a variable one)
            tmp_param_list = []     # this is an auxiliary list, to collect all the strain-related params and 
                                    # append them at the end.
            
            # parameters of the 1st Gaussian
            p_g1 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g1)

            # definition of the 1st Gaussian (note that the state variables are normalized!)
            g1 = p_g1[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g1[1])**2/(2*p_g1[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g1[2])**2/(2*p_g1[4]**2))) + p_g1[5]

            # parameters of the 2nd Gaussian
            p_g2 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g2)

            # definition of the 2nd Gaussian (note that the state variables are normalized!)
            g2 = p_g2[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g2[1])**2/(2*p_g2[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g2[2])**2/(2*p_g2[4]**2))) + p_g2[5]

            # parameters of the 3rd Gaussian
            p_g3 = self.opti.parameter(self.num_params_gaussian)    # the order is [amplitude, x0, y0, sigma_x, sigma_y, offset]
            tmp_param_list.append(p_g3)

            # definition of the 3rd Gaussian (note that the state variables are normalized!)
            g3 = p_g3[0] * np.exp(-((self.x[0]/np.max(np.abs(self.pe_boundaries))-p_g3[1])**2/(2*p_g3[3]**2) + (self.x[2]/np.max(np.abs(self.se_boundaries))-p_g3[2])**2/(2*p_g3[4]**2))) + p_g3[5]

            # definition of the symbolic cumulative strain map
            strainmap = g1 + g2 + g3
            # strainmap_sym = ca.Function('strainmap_sym', [self.x], [strainmap], {"allow_free":True})
            strainmap_sym = ca.Function('strainmap_sym', [self.x, p_g1, p_g2, p_g3], [strainmap])

            # save the symbolic strain map for some debugging
            self.strainmap_sym = strainmap_sym

        #  "Lift" initial conditions
        Xk = self.opti.variable(self.dim_x)
        init_state = self.opti.parameter(self.dim_x)     # parametrize initial condition
        self.params_list.append(init_state)              # add the parameter to the list of parameters for the NLP
        self.opti.subject_to(Xk==init_state)

        # parametrize goal position, and state describing axial rotation (phi and phi_dot)
        desired_state = self.opti.parameter(self.dim_x)  # parametrize desired goal
        self.params_list.append(desired_state)           # add the parameter to the list of parameters for the NLP

        phi_prm = self.opti.parameter(1)
        self.params_list.append(phi_prm)
        phi_dot_prm = self.opti.parameter(1)             # these are just internal parameters, not modifiable from outside

        # Collect all states/controls, and strain along the trajectory
        Xs = [Xk]
        Us = []
        Xddot_s = []
        strain_s = []

        # formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = self.opti.variable(self.dim_u)
            Us.append(Uk)
            self.opti.subject_to(u_min <= Uk)
            self.opti.subject_to(Uk <= u_max)

            # optimization variable (state) at collocation points
            Xc = self.opti.variable(self.dim_x, self.pol_order)

            # we need to create a proper input to the casadi function
            # we want to evaluate x_dot at every point in the collocation mesh
            input_sys_dynamics = ca.vertcat(Xc[0:4, :],                     # theta, theta dot, psi, psi_dot at collocation points
                                            ca.repmat(phi_prm, 1, 3),       # phi at collocation points
                                            ca.repmat(phi_dot_prm, 1, 3),   # phi_dot at collocation points
                                            ca.repmat(Uk, 1, 3),            # controls at collocation points (constant)
                                            ca.repmat(0, 1, 3))             # torque on axial rotation (zero, and constant)

            # evaluate ODE right-hand-side at collocation points
            ode = self.sys_dynamics(input_sys_dynamics)

            quad = self.cost_function(Xc, Uk, init_state, desired_state)

            # add contribution to quadrature function
            J = J + self.h*ca.mtimes(quad, self.B)

            # add acceleration values to the cost
            J = J + self.gamma_acceleration * (self.h*(ca.mtimes(ode[1, :]**2, self.B))+self.h*(ca.mtimes(ode[3,:]**2, self.B)))

            # add velocity values to the cost
            J = J + self.gamma_velocities * (self.h*(ca.mtimes(ode[0, :]**2, self.B))+self.h*(ca.mtimes(ode[2,:]**2, self.B)))
            
            # check if we care about the strain
            if self.num_gaussians>0:
                # if so, add term related to current strain to the cost
                # the strain is evaluated only at the knots of the optimization mesh
                # (note that the strain map is defined in degrees, so we convert our state to that)
                J = J + self.gamma_strain * strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3)

                # record the strain level in the current state value
                # (note that the strain map is defined in degrees, so we convert our state to that)
                strain_s.append(strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3))

            # get interpolating points of collocation polynomial
            Z = ca.horzcat(Xk, Xc)

            # get slope of interpolating polynomial (normalized)
            Pidot = ca.mtimes(Z, self.C)

            # match with ODE right-hand-side
            self.opti.subject_to(Pidot[0:4,:]==self.h*ode[0:4,:])    # theta, theta_dot, psi, psi_dot (no constraint phi)

            # save coordinates' accelerations (only for the first collocation point)
            Xddot_s.append(ode[1::2, 0])

            # state at the end of collocation interval
            Xk_end = ca.mtimes(Z, self.D)

            # new decision variable for state at the end of interval
            Xk = self.opti.variable(self.dim_x)
            Xs.append(Xk)

            # continuity constraint
            self.opti.subject_to(Xk_end==Xk)

        if self.num_gaussians>0:
            # record the strain level at the final step
            # (note that the strain map is defined in degrees, so we convert our state to that)
            strain_s.append(strainmap_sym(Xk*180/ca.pi, p_g1, p_g2, p_g3))
            self.strain_s = ca.vertcat(*strain_s)

        # adding constraint to reach the final desired state
        if self.constrain_final_vel:
            self.opti.subject_to((Xk[1]-desired_state[1])**2<self.eps_final_vel)   # bound on final (zero) velocity
            self.opti.subject_to((Xk[3]-desired_state[3])**2<self.eps_final_vel)   # bound on final (zero) velocity
        if self.constrain_final_pos:
            self.opti.subject_to((Xk[0]-desired_state[0])**2<self.eps_final_pos)   # bound on final (zero) velocity
            self.opti.subject_to((Xk[2]-desired_state[2])**2<self.eps_final_pos)   # bound on final (zero) velocity

        self.Us = ca.vertcat(*Us)
        self.Xs = ca.vertcat(*Xs)
        self.Xddot_s = ca.vertcat(*Xddot_s)

        # explicitly provide an initial guess for the primal variables
        if initial_guess_prim_vars is not None:
            self.opti.set_initial(initial_guess_prim_vars)

        # explicitly provide an initial guess for the dual variables
        if initial_guess_dual_vars is None:
            initial_guess_dual_vars = np.zeros((self.opti.lam_g.shape))

        self.opti.set_initial(self.opti.lam_g, initial_guess_dual_vars)

        # set the values of the parameters
        self.opti.set_value(init_state, self.x_0)       # this can be changed at runtime
        self.opti.set_value(desired_state, self.goal)   # this can be changed at runtime
        self.opti.set_value(phi_prm, 0)
        self.opti.set_value(phi_dot_prm, 0)

        if self.num_gaussians>0:
            self.params_list.extend(tmp_param_list)     # append at the end the parameters for the strains

            self.opti.set_value(p_g1, self.all_params_gaussians[0:6])   # NOTE: hardcoded!
            self.opti.set_value(p_g2, self.all_params_gaussians[6:12])
            self.opti.set_value(p_g3, self.all_params_gaussians[12:18])

        # define the cost function to be minimized, and store its symbolic expression
        self.opti.minimize(J + ca.sumsqr(ca.vertcat(*self.params_list)))
        self.Js = J

        # set flag indicating process was successful
        self.nlp_is_formulated = True


    def setSolverOptions(self, solver, opts):
        """
        This function allows to set the solver and the solver options that will be used
        when solving the NLP. It sets to true the corresponding flag, so that other methods
        can operate safely.
        """
        self.opti.solver(solver, opts)
        self.solver_options_set = True


    def solveNLPOnce(self):
        """
        This function solves the NLP problem that has been formulated, assuming that the constraints,
        the initial position, the goal, the cost function and the solver have been specified already.
        It retrieves the optimal trajectory for the state and control variables using the symbolic 
        mappings computes in formulateNLP(), storing those variable and returning them to the caller 
        as well.
        """
        # change the flag so that others know that the current solution is not up-to-date
        self.solution_is_updated = False

        if self.nlp_is_formulated == False:
            RuntimeError("The NLP problem has not been formulated yet! \
                         Do so with the formulateNLP() function")
            
        if self.solver_options_set == False:
            print("No user-provided solver options. \
                  Default solver options will be used. You can provide yours with setSolverOptions()")
            self.setSolverOptions(self.default_solver, self.default_solver_opts)

        self.solution = self.opti.solve()

        self.x_opt = self.solution.value(self.Xs)
        self.u_opt = self.solution.value(self.Us)
        self.J_opt = self.solution.value(self.Js)
        self.lam_g0 = self.solution.value(self.opti.lam_g)
        self.xddot_opt = self.solution.value(self.Xddot_s)

        # change the flag so that others know that the current solution is up to date
        self.solution_is_updated = True

        return self.x_opt, self.u_opt, self.solution
    
    def createOptimalMapWithoutInitialGuesses(self):
        """
        Provides a utility to retrieve a CasADi function out of an opti object, once the NLP stucture 
        has been formulated. It does formally not require solving the NLP problem beforehand.
        However, you should first run an instance of solveNLPonce() so that a good initial guess for
        primal and dual variables for the problem are used - this should speed up the solver massively.
        The function that will be generated can be used as follows (adapting it to your case):

        numerical_outputs_list = MPC_iter(numerical_values_for_parameters)

        The generated function does not allow warm-starting it.
        """

        if self.nlp_is_formulated == False:
            RuntimeError("Unable to continue. The NLP problem has not been formulated yet \
                         Do so with formulateNLP()!")
        
        if self.num_gaussians>0:
            symbolic_output_list = [self.Us, self.Xs, self.opti.lam_g, self.Js, self.strain_s, self.Xddot_s]  
        else:
            symbolic_output_list = [self.Us, self.Xs, self.opti.lam_g, self.Js, self.Xddot_s] 

        # inputs to the function
        input_list = self.params_list.copy()       # the parameters that are needed when building the NLP

        MPC_iter = self.opti.to_function('MPC_iter', input_list, symbolic_output_list)
        return MPC_iter

    def createOptimalMapInitialGuesses(self):
        """
        Provides a utility to retrieve a CasADi function out of an opti object, once the NLP stucture 
        has been formulated. It does formally not require solving the NLP problem beforehand.
        However, you should first run an instance of solveNLPonce() so that a good initial guess for
        primal and dual variables for the problem are used - this should speed up the solver massively.
        The function that will be generated can be used as follows (adapting it to your case):

        numerical_outputs_list = MPC_iter([numerical_values_for_parameters, init_guess_prim, init_guess_dual])

        The generated function needs as inputs the initial guesses for both primal and dual variables.
        """

        if self.nlp_is_formulated == False:
            RuntimeError("Unable to continue. The NLP problem has not been formulated yet \
                         Do so with formulateNLP()!")
            
        # inputs to the function
        input_list = self.params_list.copy()       # the parameters that are needed when building the NLP
        if self.solution is None:
            RuntimeError('No inital guess can be used for primal variables!\
                         Run solveNLPonce() first. \n')
        else:
            input_list.append(self.Us)       # the initial guess for the controls
            input_list.append(self.Xs)       # the initial guess for the state trajectory

        if self.lam_g0 is None:
            RuntimeError('No inital guess can be used dual variables! \
                         Run solveNLPonce() first \n')
        else:
            input_list.append(self.opti.lam_g)  # the inital guess for the dual variable

        if self.num_gaussians>0:
            symbolic_output_list = [self.Us, self.Xs, self.opti.lam_g, self.Js, self.strain_s, self.Xddot_s]
        else:
            symbolic_output_list = [self.Us, self.Xs, self.opti.lam_g, self.Js, self.Xddot_s]

        MPC_iter = self.opti.to_function('MPC_iter', input_list, symbolic_output_list)
        return MPC_iter
    

    def getSizePrimalVars(self):
        """
        This function allows to retrieve the dimension of the primal variables of the problem, after it 
        has been solved at least once
        """
        if self.Xs is None or self.Us is None:
            RuntimeError('No stored values for primal variables!\
                         Run solveNLPonce() first. \n')
        else:
            return (self.Xs.shape, self.Us.shape)
    

    def getSizeDualVars(self):
        """
        This function allows to retrieve the dimension of the dual variables of the problem, after it 
        has been solved at least once
        """

        if self.lam_g0 is None:
            RuntimeError('No stored values for dual variables! \
                         Run solveNLPonce() first \n')
        else:
            return np.shape(self.lam_g0)
        

    def visualizeCurrentStrainMap(self, threeD = False, block = False):
        """
        Call this function to display the strain map currently considered by the NLPS.
        Select whether you want to visualize it in 3D or not.
        """
        # inizialize empty strain map
        current_strainmap = np.zeros(self.X_norm.shape)

        # loop through all of the Gaussians, and obtain the strain map values
        for function in range(len(self.all_params_gaussians)//self.num_params_gaussian):

            #first, retrieve explicitly the parameters of the function considered in this iteration
            amplitude = self.all_params_gaussians[function*self.num_params_gaussian]
            x0 = self.all_params_gaussians[function*self.num_params_gaussian+1]
            y0 = self.all_params_gaussians[function*self.num_params_gaussian+2]
            sigma_x = self.all_params_gaussians[function*self.num_params_gaussian+3]
            sigma_y = self.all_params_gaussians[function*self.num_params_gaussian+4]
            offset = self.all_params_gaussians[function*self.num_params_gaussian+5]
            
            # then, compute the contribution of this particular Gaussian to the final strain map
            current_strainmap += amplitude * np.exp(-((self.X_norm-x0)**2/(2*sigma_x**2)+(self.Y_norm-y0)**2/(2*sigma_y**2)))+offset
            
        # finally, plot the resulting current strain map
        if threeD:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(self.X, self.Y, current_strainmap, cmap='plasma')
            ax.set_xlabel('Plane Elev [deg]')
            ax.set_ylabel('Shoulder Elev [deg]')
            ax.set_zlabel('Strain level [%]')
            ax.set_zlim([0, current_strainmap.max()])
        else:
            fig = plt.figure()
            ax = fig.add_subplot()
            heatmap = ax.imshow(np.flip(current_strainmap.T, axis = 0), cmap='plasma', extent=[self.X.min(), self.X.max(), self.Y.min(), self.Y.max()])
            fig.colorbar(heatmap, ax = ax, ticks=np.arange(0, current_strainmap.max() + 1), label='Strain level [%]')
            ax.set_xlabel('Plane Elev [deg]')
            ax.set_ylabel('Shoulder Elev [deg]')

        plt.show(block=block)


    def visualize2DTrajectoryOnCurrentStrainMap(self, trajectory_2d, strains, threeD):
        """
        This function allows to plot a given trajectory (in the plane of elevation-shoulder elevation space)
        on the strain map that is currently considered. The inputs are:
        - trajectory_2d: sequence of (pe, se) values defining the trajectory in shoulder space [in radians]
                         Note: axial rotation is not considered since it is fixed on a given strain map.
        - strains: sequence of the strains associated to each point in the trajectory.
        - threeD: whether you want to visualize it in 3D or not.

        Note: the length of the trajectory should be coherent with the amount of strain points given
        """
        # check if the input are consistent
        assert trajectory_2d.shape[1] == strains.shape[1], "The number of points in the trajectory should correspond \
                                                            to the strain data-points"
        
        assert trajectory_2d.shape[0] == 2, "The given trajectory must be 2-dimensional \
                                             (points must be [plane of elevation, shoulder elevation])"
        
        # inizialize empty strain map
        current_strainmap = np.zeros(self.X_norm.shape)

        # loop through all of the Gaussians, and obtain the strain map values
        for function in range(len(self.all_params_gaussians)//self.num_params_gaussian):

            #first, retrieve explicitly the parameters of the function considered in this iteration
            amplitude = self.all_params_gaussians[function*self.num_params_gaussian]
            x0 = self.all_params_gaussians[function*self.num_params_gaussian+1]
            y0 = self.all_params_gaussians[function*self.num_params_gaussian+2]
            sigma_x = self.all_params_gaussians[function*self.num_params_gaussian+3]
            sigma_y = self.all_params_gaussians[function*self.num_params_gaussian+4]
            offset = self.all_params_gaussians[function*self.num_params_gaussian+5]
            
            # then, compute the contribution of this particular Gaussian to the final strain map
            current_strainmap += amplitude * np.exp(-((self.X_norm-x0)**2/(2*sigma_x**2)+(self.Y_norm-y0)**2/(2*sigma_y**2)))+offset
            
        # finally, plot the resulting current strain map together with the trajectory (and its associated strain)
        strain_at_the_goal = self.strainmap_sym(self.goal*180/np.pi, self.all_params_gaussians[0:6], self.all_params_gaussians[6:12], self.all_params_gaussians[12:18])
        
        if threeD:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(self.X, self.Y, current_strainmap, cmap='plasma', alpha=0.5)
            ax.set_xlabel('Plane Elev [deg]')
            ax.set_ylabel('Shoulder Elev [deg]')
            ax.set_zlabel('Strain level [%]')
            ax.set_zlim([0, current_strainmap.max()])
            ax.plot3D(np.rad2deg(trajectory_2d[0, :]), np.rad2deg(trajectory_2d[1, :]), strains.squeeze(), '-', color='black')
            ax.scatter3D(np.rad2deg(self.goal[0]), np.rad2deg(self.goal[2]), strain_at_the_goal, '-', color='blue', label = 'goal position')
            ax.scatter3D(np.rad2deg(trajectory_2d[0, 0]), np.rad2deg(trajectory_2d[1, 0]), strains[0,0], '-', color='green', label='initial position')
        else: 
            fig = plt.figure()
            ax = fig.add_subplot()
            heatmap = ax.imshow(np.flip(current_strainmap.T, axis = 0), cmap='plasma', extent=[self.X.min(), self.X.max(), self.Y.min(), self.Y.max()])
            fig.colorbar(heatmap, ax = ax, ticks=np.arange(0, current_strainmap.max() + 1), label='Strain level [%]')
            ax.set_xlabel('Plane Elev [deg]')
            ax.set_ylabel('Shoulder Elev [deg]')
            ax.plot(np.rad2deg(trajectory_2d[0, :]), np.rad2deg(trajectory_2d[1, :]), '-', color='black')
            ax.scatter(np.rad2deg(self.goal[0]), np.rad2deg(self.goal[2]), color='blue', label = 'goal position')
            ax.scatter(np.rad2deg(trajectory_2d[0, 0]), np.rad2deg(trajectory_2d[1, 0]), color='green', label='initial position')
        
        plt.show(block=False)


    def visualizeCurrentSolution(self, viz_state = True, viz_controls = True, blocking = False, save = False, path_save = ''):
        """
        This function allows to visualize the current solution in terms of state trajectory and control values

        """
        fig, axes = plt.subplots(1, 2 if viz_state and viz_controls else 1)

        if viz_state:
            ax = axes[0] if viz_state and viz_controls else axes
            ax.plot(self.x_opt[0, :], label='pe')
            ax.plot(self.x_opt[2, :], label='se')
            ax.plot(self.x_opt[4, :], label='ar')
            ax.set_title('State variables')
            ax.set_xlabel(f'time [* {self.h}s]')
            ax.set_ylabel('value [rad]')
            ax.legend()

        if viz_controls:
            ax = axes[1] if viz_state and viz_controls else axes
            if ax is not None:
                ax.plot(self.u_opt[0, :], label='tau_pe')
                ax.plot(self.u_opt[1, :], label='tau_se')
                ax.set_title('Control variables')
                ax.set_xlabel(f'time [* {self.h}s]')
                ax.set_ylabel('Torque [Nm]')
                ax.legend()

        # plt.tight_layout()

        plt.show(block=blocking)


    def setParametersStrainMap(self, num_gaussians, all_params_gaussians, rounded_axial_rotation, act_current = None):
        """
        Utility to set the parameters defining the strain map that is considered by the NLP.
        """
        self.num_gaussians = num_gaussians
        self.all_params_gaussians = all_params_gaussians
        self.strain_visualizer.ar_current = rounded_axial_rotation
        self.strain_visualizer.act_current = act_current


    def setCostWeights(self, goal = 0, strain = 0, velocities = 0, acceleration = 0):
        """
        Utility to set the weights for the various terms in the cost function of the NLP.
        """
        self.gamma_goal = goal                  # weight of the distance to the goal [1/(rad^2)]
        self.gamma_strain = strain              # weight of the strain value in the cost function
        self.gamma_velocities = velocities      # weight for pe_dot and se_dot
        self.gamma_acceleration = acceleration  # weight on pe_ddot and se_ddot


class RealTimeStrainMapVisualizer:
    """
    This class is built to have a way to visualize the 2D strain maps and update them in real-time
    """
    def __init__(self, X_norm, Y_norm, num_params_gaussian, pe_boundaries, se_boundaries):
        """
        Initialize the visualizer with the parameters that define
        """
        self.X_norm = X_norm
        self.Y_norm = Y_norm
        self.num_params_gaussian = num_params_gaussian
        self.ar_current = None          # current (rounded) value of the axial rotation, associated to the strain map
        self.pe_boundaries = pe_boundaries
        self.se_boundaries = se_boundaries

        self.act_current = None         # current (rounded) value of the activation

        # create a surface to hold the image of the strain map
        self.image_surface = pygame.Surface(np.shape(X_norm))

        # define the bound values for the strain
        self.min_strain = 0
        self.max_strain = 8

        # dimensions of the resulting windows in pixels
        self.widow_dimensions = (800, 600)
        self.tick_length = 5                # length of axis ticks, in pixels
        self.width_lines = 3                # width of the lines, in pixels
        self.color_lines = (255, 255, 255)

        self.font_title_pygame = None
        self.font_title = 24
        self.font_ticks_pygame = None
        self.font_ticks = 18
        self.ticks_x = None

        self.debug = 0

        if self.debug:
            # Define a custom color gradient based on the data range
            num_colors = 256  # Adjust as needed
            self.custom_palette = np.zeros((num_colors, 3), dtype=np.uint8)

            # Adjust the color gradient to match the data distribution
            data_min = 0
            data_max = 7
            for i in range(num_colors):
                # Interpolate color components based on the data range
                value = (i / (num_colors - 1))  # Normalize value to range [0, 1]
                self.custom_palette[i, 0] = 255 * (value - data_min) / (data_max - data_min)  # Red component
                self.custom_palette[i, 1] = 0  # Green component (adjust as needed)
                self.custom_palette[i, 2] = 255 * (1 - (value - data_min) / (data_max - data_min))  # Blue component

            # Convert the custom palette to a list of tuples
            self.custom_palette = [tuple(color) for color in self.custom_palette.tolist()]

        self.is_running = False


    def map_to_color(self, values):
        """
        Function to map values to colors. We want to reproduce the "hot" colormap of matplotlib.
        """ 
        # Normalize the values to the range [0, 1]
        # normalized_values = (values - self.min_strain) / (self.max_strain - self.min_strain)
        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-5)

        # Apply colormap transformation
        red = np.clip(2 * normalized_values, 0, 1)
        green = np.clip(2 * normalized_values - 1, 0, 1)
        blue = np.clip(4 * normalized_values - 3, 0, 1)
        
        # Stack color channels and convert to uint8
        colors = np.stack((red, green, blue), axis=-1) * 255
        colors = colors.astype(np.uint8)
        return colors


    def remapPointOnScreen(self, point):
        """
        Utility to remap a given point (pe,se) in the correct position on the screen.
        """
        # first, we find the position of our point as a scale factor for both the coordinates
        # essentially, if the point is very close to the upper bound of its two coordinates, this value is ~1
        # while if it is very close to the lower to the lower bound, it is ~0
        pe_scale_factor = (point[0]-self.pe_boundaries[0])/(self.pe_boundaries[1]-self.pe_boundaries[0])
        se_scale_factor = (point[1]-self.se_boundaries[0])/(self.se_boundaries[1]-self.se_boundaries[0])

        # we then use this information to map the point on the actual screen, as we know the screen size
        # mind that the origin of the screen is in the top left corner, with X from left to right, and Y downwards
        pe_on_screen = pe_scale_factor * self.widow_dimensions[0]
        se_on_screen = (1-se_scale_factor) * self.widow_dimensions[1]   # note the correction, as the strain map has SE positive
                                                                        # upwards. So we flip the position wrt the Y axis
        
        # return the re-projected point
        return np.array([pe_on_screen, se_on_screen])
    

    def draw_x_axis(self, ticks):
        pygame.draw.line(self.screen, 
                         self.color_lines, 
                         (0, self.widow_dimensions[1]), 
                         (self.widow_dimensions[0], self.widow_dimensions[1]), 
                         self.width_lines)

        for i in range(0, self.widow_dimensions[0] + 1, int(self.widow_dimensions[0]/((self.pe_boundaries[1]-self.pe_boundaries[0])//10))):
            x = i
            y = self.widow_dimensions[1] - self.width_lines
            pygame.draw.line(self.screen, self.color_lines, (x, y - self.tick_length), (x, y + self.tick_length), self.width_lines)

        # Render and blit the ticks text
        num_ticks = np.shape(ticks)[0]

        for i in range(num_ticks):
            caption_text = self.font_ticks_pygame.render(str(ticks[i]), True, self.color_lines)
            self.screen.blit(caption_text, np.array([self.remapPointOnScreen(np.array([ticks[i], 0]))[0]-2*i, self.widow_dimensions[1] - 20]))
            # the -2*i above is just to compensate for some weird drift. Might be a mistake I made as well, not sure...

    def draw_y_axis(self, ticks):
        pygame.draw.line(self.screen, 
                         self.color_lines, 
                         (0, 0), 
                         (0, self.widow_dimensions[1]), 
                         self.width_lines)

        for i in range(0, self.widow_dimensions[1] + 1, int(self.widow_dimensions[1]/((self.se_boundaries[1]-self.se_boundaries[0])//10))):
            x = self.width_lines
            y = i
            pygame.draw.line(self.screen, self.color_lines, (x - self.tick_length, y), (x + self.tick_length, y), self.width_lines)

        # Render and blit the ticks text
        num_ticks = np.shape(ticks)[0]

        for i in range(num_ticks):
            caption_text = self.font_ticks_pygame.render(str(ticks[i]), True, self.color_lines)
            self.screen.blit(caption_text, np.array([10, self.remapPointOnScreen(np.array([0, ticks[i]]))[1]-4*(num_ticks -i)]))


    def updateStrainMap(self, list_params, pose_current = None, trajectory_current = None, goal_current = None, vel_current = None):
        """
        This function allows to update the strain map.
        Inputs:
        * list_params: contains the list of parameters that define the gaussians
                       that need to be plotted.
        * pose_current: the current estimated pose on the strain map
        * trajectory current: the current optimal trajectory (as points)
        TODO: we could also add capability to plot what was considered optimal at the previous time
        step, and the position where the optimization started?
        """
        # check if this is the first time that the visualizer is used
        # if so, instantiate the window first
        if not self.is_running:
            pygame.init()
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode(self.widow_dimensions)
            # font for the caption
            self.font_title_pygame = pygame.font.Font(None, self.font_title)
            pygame.display.set_caption('Real-Time Strain Map')

            # font for the axis
            self.font_ticks_pygame = pygame.font.Font(None, self.font_ticks)

            self.is_running = True

        current_strainmap = np.zeros(self.X_norm.shape)

        for function in range(len(list_params) // self.num_params_gaussian):
            amplitude = list_params[function * self.num_params_gaussian]
            x0 = list_params[function * self.num_params_gaussian + 1]
            y0 = list_params[function * self.num_params_gaussian + 2]
            sigma_x = list_params[function * self.num_params_gaussian + 3]
            sigma_y = list_params[function * self.num_params_gaussian + 4]
            offset = list_params[function * self.num_params_gaussian + 5]

            current_strainmap += amplitude * np.exp(
                -((self.X_norm - x0) ** 2 / (2 * sigma_x ** 2) + (self.Y_norm - y0) ** 2 / (2 * sigma_y ** 2))) + offset

        # Map values to colors directly using numpy array indexing
        colors = self.map_to_color(current_strainmap)

        # Set the entire surface with colors
        pygame.surfarray.blit_array(self.image_surface, np.flip(colors, axis=1))  # Transpose to match pygame surface format

        self.screen.blit(pygame.transform.scale(self.image_surface, self.widow_dimensions), (0, 0))

        # Render and blit the caption
        ar_label = self.font_title_pygame.render(f'Axial rotation:{self.ar_current}', True, self.color_lines)
        self.screen.blit(ar_label, (self.widow_dimensions[0]-200, 10))

        act_label = self.font_title_pygame.render(f'Muscle activation:{self.act_current}', True, self.color_lines)
        self.screen.blit(act_label, (self.widow_dimensions[0]-200, 40))

        # if given, display the current 2D pose on the strain map (plane of elevation, shoulder elevation)
        if pose_current is not None:
            marker_radius = 5      # define the radius of the marker for the current pose
            pygame.draw.circle(self.screen, (255, 0, 0), self.remapPointOnScreen(np.rad2deg(pose_current)), marker_radius)

        if vel_current is not None:
            pygame.draw.line(self.screen, 
                             (255, 0, 0), 
                             self.remapPointOnScreen(np.rad2deg(pose_current)), 
                             self.remapPointOnScreen(np.rad2deg(pose_current+vel_current)), 
                             self.width_lines)

        # if given, display the reference trajectory scattering its points
        if trajectory_current is not None:
            traj_point_radius = 3
            
            for index in range(np.shape(trajectory_current)[1]):
                pygame.draw.circle(self.screen, (0, 0, 255), self.remapPointOnScreen(np.rad2deg(trajectory_current[:,index])), traj_point_radius)

        # visualize also the goal on the current strain map (if it has been set)
        if goal_current is not None:
            goal_radius = 5
            pygame.draw.circle(self.screen, (0, 255, 0), self.remapPointOnScreen(np.rad2deg(goal_current)), goal_radius)

        # draw the X and Y axis on the map
        self.draw_x_axis(np.array([0, 40, 80, 120, 140]))
        self.draw_y_axis(np.array([20, 40, 80, 140]))

        pygame.display.flip()


    def quit(self):
        pygame.quit()



def is_hsl_present():
    lib_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    for path in lib_paths:
        if os.path.exists(os.path.join(path, "libhsl.so")):
            return True
    try:
        ctypes.CDLL("libhsl.so")  # Try loading the shared library
        return True
    except OSError:
        return False