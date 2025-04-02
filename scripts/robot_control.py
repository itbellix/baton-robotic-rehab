"""
Script to control the KUKA LBR iiwa both in simulation and in the lab.
It builds on top of the iiwa_impedance_control repository, available at https://gitlab.tudelft.nl/nickymol/iiwa_impedance_control/

Example usage: $ python robot_control --simulation=true
"""

# Used in the main computations
import numpy as np

# ROS libraries
import rospy # CC
import actionlib
from controller_manager_msgs.srv import SwitchController

# Messages to and from KUKA robot
from iiwa_impedance_control.msg import CartesianTrajectoryExecutionAction, CartesianTrajectoryExecutionGoal, \
                                JointTrajectoryExecutionAction, JointTrajectoryExecutionGoal, JointTrajectoryExecutionActionResult
from std_msgs.msg import Float64MultiArray, Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState

from spatialmath import SE3
import math

# import dynamic reconfigure client to change parameters of the controller
import dynamic_reconfigure.client

# import scipy to deal with rotation matrices
from scipy.spatial.transform import Rotation as R
import time

# import tkinter to be used as an interface for executing code in a state-machine fashion
import tkinter as tk

# import threading for testing
import threading

# import parser
import argparse

# import the parameters for the experiment as defined in experiment_parameters.py
from experiment_parameters import *

# import the robot model
import lbr_iiwa_robot_model as lbr

class RobotControlModule:
    """
    This class implements the robot control
    """
    def __init__(self, shared_ros_topics, experimental_params):
        """
        Initializes a RobotControlModule object, given:
        * shared_ros_topics: list of names of ROS topics used to communicate with a TO_module
        * experimental_params: parameters that are dependent on the experimental setup 
                               (such as arm length, orientation wrt the robot, etc...)
        """
        # define the robot model
        self.robot = lbr.LBR7_iiwa_ros_DH()

        # Action clients
        self.cartesian_action_client = actionlib.SimpleActionClient('/CartesianImpedanceController/cartesian_trajectory_execution_action', CartesianTrajectoryExecutionAction)
        self.joint_action_client = actionlib.SimpleActionClient('/JointImpedanceController/joint_trajectory_execution_action', JointTrajectoryExecutionAction)

        # Wait for servers
        try:
            rospy.loginfo("Waiting for cartesian_trajectory_execution action server...")
            self.cartesian_action_client.wait_for_server()
            rospy.loginfo(f"cartesian_trajectory_execution server found!")
        except:
            rospy.logwarn("cartesian_trajectory_execution action server not found...", "warn")
        try:
            rospy.loginfo("Waiting for joint_trajectory_execution action server...")
            self.joint_action_client.wait_for_server()
            rospy.loginfo(f"joint_trajectory_execution server found!")
        except:
            rospy.logwarn("joint_trajectory_execution server not found...", "warn")

        # Controller manager service
        try:
            rospy.wait_for_service('/iiwa/controller_manager/switch_controller', timeout=10)
            self.controller_manager = rospy.ServiceProxy('/iiwa/controller_manager/switch_controller', SwitchController)
            rospy.loginfo("Controller manager found!")
        except rospy.ROSException:
            rospy.logwarn("Controller manager not found!")

        # keep track of the controller currently used
        self.active_controller = 'JIC'     # options are "JIC" or "CIC" (Joint Impedance Controller or Cartesian Impedance Controller)

        # instantiate a client to modify the parameters of the controller in real-time
        self.reconf_client_cart = dynamic_reconfigure.client.Client('/CartesianImpedanceController/dynamic_reconfigure_server_node', timeout=30)
        self.reconf_client_joint = dynamic_reconfigure.client.Client('/JointImpedanceController/dynamic_reconfigure_server_node', timeout=30)

        # define ROS subscribers
        self.sub_to_cartesian_pose = rospy.Subscriber('/CartesianImpedanceController/cartesian_pose',   # name of the topic to subscribe to
                                                      PoseStamped,                                      # type of ROS message to receive
                                                      self._callback_ee_pose,                           # callback
                                                      queue_size=1)
        
        self.sub_to_cartesian_twist = rospy.Subscriber('/CartesianImpedanceController/cartesian_twist',
                                                        TwistStamped,
                                                        self._callback_ee_twist,
                                                        queue_size=1)
        
        self.sub_to_joint_state_cart = rospy.Subscriber('/CartesianImpedanceController/joint_states',
                                                   JointState,
                                                   self._callback_joint_state_cart,
                                                   queue_size=1)
        
        self.sub_to_joint_state_joint = rospy.Subscriber('/JointImpedanceController/joint_states',
                                                   JointState,
                                                   self._callback_joint_state_joint,
                                                   queue_size=1)
        
        # define a ROS publisher to convert current cartesian pose into shoulder pose
        self.topic_shoulder_pose= shared_ros_topics['estimated_shoulder_pose']
        self.pub_shoulder_pose = rospy.Publisher(self.topic_shoulder_pose, Float32MultiArray, queue_size = 1)

        # define a ROS subscriber to receive the commanded (optimal) trajectory for the EE, from the
        # biomechanical-based optimization
        self.topic_optimal_pose_ee = shared_ros_topics['optimal_cartesian_ref_ee']
        self.sub_to_optimal_pose_ee = rospy.Subscriber(self.topic_optimal_pose_ee, 
                                                       PoseStamped,
                                                       self._callback_ee_optimal_pose,
                                                       queue_size=1)
        
        # define the parameters that will be used to store information about the robot status and state estimation
        self.ee_pose_curr = None            # EE current pose
        self.ee_desired_pose = None         # EE desired pose
        self.ee_twist_curr = None           # EE current twist (linear and angular velocity in robot base frame)
        self.desired_pose_reached = False    # the end effector has effectively reached the desired point
        self.initial_pose_reached = False   # one-time flag to be adjusted when the robot reaches the required
                                            # starting point. When set to true, the estimated shoulder state are meaningful
        self.joint_states = None
        self.joint_velocities = None
        self.joint_efforts = None

        # define parameters for the filtering of the human state estimation
        self.alpha_p = 0.8                  # weight of the exponential moving average filter (position part)
        self.alpha_v = 0.8                  # weight of the exponential moving average filter (velocity part)
        self.alpha_a = 1.0                  # weight of the exponential moving average filter (acceleration part)
        self.human_pose_estimated = np.zeros(9)
        self.last_timestamp_estimation = None       # timestamp of the last human pose estimation
        self.filter_initialized = False     # whether the filter applied on the human pose estimation has
                                            # already been initialized
        
        # set up a publisher and its thread for publishing continuously whether the robot is tracking the 
        # optimal trajectory or not
        self.topic_request_reference = shared_ros_topics['request_reference']
        self.pub_request_reference = rospy.Publisher(self.topic_request_reference, Bool, queue_size = 1)
        self.requesting_reference = False         # flag indicating whether the robot is requesting a reference
        self.thread_therapy_status = threading.Thread(target=self.requestUpdatedReference)   # creating the thread
        self.thread_therapy_status.daemon = True    # this allows to terminate the thread when the main program ends

        # store the experimental parameters
        self.exp_prms = experimental_params


    def _callback_ee_twist(self,data):
        """
        This callback is dedicated to listening to the current twist of the end effector.
        Internal parameters of the module are updated accordingly (v and omega in robot's base frame).
        """
        self.ee_twist_curr = np.array([data.twist.linear.x, data.twist.linear.y, data.twist.linear.z,
                                       data.twist.angular.x, data.twist.angular.y, data.twist.angular.z])
        

    def _callback_joint_state_cart(self,data):
        """
        This callback is dedicated to listening to the current joint state of the robot.
        Internal parameters of the module are updated accordingly. This is triggered when the Cartesian impedance controller is active.
        """
        self.joint_states = np.array(data.position)
        self.joint_velocities = np.array(data.velocity)
        self.joint_effors = np.array(data.effort)

    
    def _callback_joint_state_joint(self,data):
        """
        This callback is dedicated to listening to the current joint state of the robot.
        Internal parameters of the module are updated accordingly. This is triggered when the Joint impedance controller is active.
        """
        self.joint_states = np.array(data.position)
        self.joint_velocities = np.array(data.velocity)
        self.joint_effors = np.array(data.effort)


    def _callback_ee_pose(self,data):
        """
        This callback is linked to the ROS subscriber that listens to the topic where the cartesian pose and twist of the EE is published.
        Message type is expected to be custom PoseTwistStamped, defined in the iiwa_impedance_control package available at https://gitlab.tudelft.nl/nickymol/iiwa_impedance_control.
        It processes the data received and updates the internal parameters of the RobotControlModule accordingly.
        If the desired position for the therapy/experiment has been reached, it also converts the current EE pose into shoulder pose 
        (under the assumption that the glenohumeral joint center is fixed in space), and publishes this information on a topic.
        """
        timestamp_msg = data.header.stamp.to_time()
        cart_pose_ee = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        orientation_quat = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[0:3, 0:3] = R.from_quat(orientation_quat).as_matrix()
        homogeneous_matrix[0:3, 3] = cart_pose_ee
        self.ee_pose_curr = SE3(homogeneous_matrix)   # value for the homogenous matrix describing ee pose
        
        # check if the robot has already reached its desired pose and if at least one twist message has been received 
        # If so, estimate and publish human (shoulder) poses
        if self.initial_pose_reached and self.ee_twist_curr is not None:
            R_ee = self.ee_pose_curr.R    # retrieve the rotation matrix defining orientation of EE frame
            sh_R_elb = np.transpose(experimental_params['base_R_shoulder'].as_matrix())@R_ee@np.transpose(experimental_params['elb_R_ee'].as_matrix())

            direction_vector = cart_pose_ee - experimental_params['p_gh_in_base']
            direction_vector_norm = direction_vector / np.linalg.norm(direction_vector)

            direction_vector_norm_in_shoulder = np.transpose(experimental_params['base_R_shoulder'].as_matrix())@direction_vector_norm

            # 1. we estimate the coordinate values
            # The rotation matrix expressing the elbow frame in shoulder frame is approximated as:
            # sh_R_elb = R_y(pe) * R_x(-se) * R_y (ar) = 
            # 
            #   | c_pe*c_ar-s_pe*c_se*s_ar    -s_pe*s_se    c_pe*s_ar+s_pe*c_se*c_ar  |
            # = |          -s_ar                 c_se               s_se*c_ar         |
            #   | -s_pe*c_ar-c_pe*c_se*s_ar   -c_pe*s_se    -s_pe*s_ar+c_pe*c_se*c_ar |
            #
            # Thus, the following formulas retrieve the shoulder state:
            
            # estimation of {pe,se} based on the cartesian position of the EE
            pe = np.arctan2(direction_vector_norm_in_shoulder[0], direction_vector_norm_in_shoulder[2])
            se = np.arccos(np.dot(direction_vector_norm_in_shoulder, np.array([0, -1, 0])))
            
            # estimation of {pe,se} based on the orientation of the EE
            # se = np.arctan2(np.sqrt(sh_R_elb[0,1]**2+sh_R_elb[2,1]**2), sh_R_elb[1,1])
            # pe = np.arctan2(-sh_R_elb[0,1]/np.sin(se), -sh_R_elb[2,1]/np.sin(se))

            # once the previous angles have been retrieved, use the orientation of the EE to find ar
            # (we assume that the EE orientation points towards the center of the shoulder for this)
            ar = np.arctan2(-sh_R_elb[1,0], sh_R_elb[1,2]/np.sin(se)) + experimental_params['ar_offset']

            # 2. we estimate the coordinate velocities (here the robot and the human are interacting as a geared mechanism)
            # 2.1 estimate the velocity along the plane of elevation
            sh_twist = experimental_params['base_R_shoulder'].as_matrix().T @ self.ee_twist_curr.reshape((2,3)).T
            r = np.array([experimental_params['L_tot'] * np.cos(pe), experimental_params['L_tot'] * np.sin(pe)])

            # calculating the angular velocity around the Y axis of the shoulder frame (pointing upwards)
            # formula : omega = radius_vec X velocity_vec / (||radius_vec||^2)
            # velocities and radius are considered on the plane perpendicular to Y (so, the Z-X plane)
            pe_dot = np.cross(r, np.array([sh_twist[2,0], sh_twist[0,0]]))/(experimental_params['L_tot']**2)

            # 2.2. estimate the velocity along the shoulder elevation
            # First transform the twist in the local frame where this DoF is defined, then apply the same
            # formula as above to obtain angular velocity given the linear speed and distance from the rotation axis
            # Note the minus sign in front of the cross-product, for consistency with the model definition.
            local_twist = R.from_euler('y', pe).as_matrix().T @ sh_twist
            r = np.array([experimental_params['L_tot'] * np.sin(se), -experimental_params['L_tot'] * np.cos(se)])
            se_dot = np.cross(r, np.array([local_twist[2,0], local_twist[1,0]]))/(experimental_params['L_tot']**2)

            # 2.3 estimate the velocity along the axial rotation
            elb_twist = R.from_euler('x', -se).as_matrix().T @ local_twist
            ar_dot = elb_twist[1,1]

            # filter the state values (we use an exponential filter)
            if not self.filter_initialized:
                # if the filter state has not been initialized yet, do it now
                self.human_pose_estimated[0] = np.round(pe, 3)
                self.human_pose_estimated[1] = np.round(pe_dot, 3)
                self.human_pose_estimated[2] = np.round(se, 3)
                self.human_pose_estimated[3] = np.round(se_dot, 3)
                self.human_pose_estimated[4] = np.round(ar, 3)
                self.human_pose_estimated[5] = np.round(ar_dot, 3)
                self.human_pose_estimated[6] = 0            # we initialize the accelerations to be 0
                self.human_pose_estimated[7] = 0            # we initialize the accelerations to be 0
                self.human_pose_estimated[8] = 0            # we initialize the accelerations to be 0

                self.filter_initialized = True

            # filter the estimates for the human pose (glenohumeral angles and velocities)    
            pe = self.alpha_p * pe + (1-self.alpha_p) * self.human_pose_estimated[0]
            pe_dot = self.alpha_v * pe_dot + (1-self.alpha_v) * self.human_pose_estimated[1]
            se = self.alpha_p * se + (1-self.alpha_p) * self.human_pose_estimated[2]
            se_dot = self.alpha_v * se_dot + (1-self.alpha_v) * self.human_pose_estimated[3]
            ar = self.alpha_p * ar + (1-self.alpha_p) * self.human_pose_estimated[4]
            ar_dot = self.alpha_v * ar_dot + (1-self.alpha_v) * self.human_pose_estimated[5]

            # retrieve accelerations differentiating numerically the velocities
            pe_ddot = 0
            se_ddot = 0
            ar_ddot = 0

            if self.last_timestamp_estimation is not None:
                delta_t = timestamp_msg - self.last_timestamp_estimation
                if delta_t > 0:
                    pe_ddot = (pe_dot - self.human_pose_estimated[1])/delta_t
                    se_ddot = (se_dot - self.human_pose_estimated[3])/delta_t
                    ar_ddot = (ar_dot - self.human_pose_estimated[5])/delta_t

                    # filter accelerations too
                    # note that here self.alpha_a=1
                    pe_ddot = self.alpha_a * pe_ddot + (1-self.alpha_a) * self.human_pose_estimated[6]
                    se_ddot = self.alpha_a * se_ddot + (1-self.alpha_a) * self.human_pose_estimated[7]
                    ar_ddot = self.alpha_a * ar_ddot + (1-self.alpha_a) * self.human_pose_estimated[8]

            if math.isnan(self.human_pose_estimated[8]):
                aux = 0

            # update last estimated pose (used as state of the filter)
            self.human_pose_estimated[0] = np.round(pe, 3)
            self.human_pose_estimated[1] = np.round(pe_dot, 3)
            self.human_pose_estimated[2] = np.round(se, 3)
            self.human_pose_estimated[3] = np.round(se_dot, 3)
            self.human_pose_estimated[4] = np.round(ar, 3)
            self.human_pose_estimated[5] = np.round(ar_dot, 3)
            self.human_pose_estimated[6] = np.round(pe_ddot, 3)
            self.human_pose_estimated[7] = np.round(se_ddot, 3)
            self.human_pose_estimated[8] = np.round(ar_ddot, 3)

            self.last_timestamp_estimation = timestamp_msg      # used as the timestamp of the previous information

            # build the message and fill it with information
            message = Float32MultiArray()
            message.data = np.round(np.concatenate((self.human_pose_estimated, orientation_quat, cart_pose_ee)), 3)

            # publish only if ROSCORE is running
            if not rospy.is_shutdown():
                self.pub_shoulder_pose.publish(message)


    def _callback_ee_optimal_pose(self,data):
        """
        This callback is linked to the ROS subscriber that listens to the topic where the desired cartesian pose of the EE 
        is published. It processes the data received (PoseStamped) and updates the internal parameters of the RobotControlModule.
        """
        # the message that we receive contains the required cartesian pose for the end effector in terms of an homogenous matrix.
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[0:3, 0:3] = R.from_quat([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]).as_matrix()
        homogeneous_matrix[0:3, 3] = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.ee_desired_pose = homogeneous_matrix

        # calculate corresponding 3D position and euler angles just in case
        # xyz_position = homogeneous_matrix[0:3, 3]
        # euler_angles = R.from_matrix(homogeneous_matrix[0:3, 0:3]).as_euler('zxy', degrees=False)

    
    def togglePublishingEEPose(self, flag, last_controller_mode=None):
        """
        This function allows to enable/disable the stream of data related to the position of the end-effector in cartesian
        space (expressed in the frame centered at the robot base).
        The inputs are:
            - flag: can be set to True/False if we want to start/stop publishing the ee pose
            - last_controller_mode: it is a ControllerGoal object that represents the one that is being used to 
                                    control the robot when this function is called. The same goal is sent again 
                                    at the end of the function call, so that the robot will maintain the position,
                                    stiffness and damping that it had before. If it is None, a default cartesian
                                    impedance controller will be used instead to keep the current position.

        Note: the execution of this function might be a bit slow. This should not be a problem as it is meant to 
        be used at the beginning of the experiment (otherwise, make sure that it does not cause delays).
        """

        # now we can actually send what we wanted without worrying 
        self.cartesian_pose_publisher.flag = flag
        self.client.wait_for_server()
        self.client.send_goal(self.cartesian_pose_publisher)
        self.client.wait_for_result()

        if last_controller_mode is not None:
            self.client.wait_for_server()
            self.client.send_goal(last_controller_mode)
            self.client.wait_for_result()


    def getEEDesiredPose(self):
        """
        Utility function to retrieve the value of the end effector desired pose stored in the RobotControlModule
        """
        return self.ee_desired_pose

    
    def moveToEEDesiredPose(self, time, rate, precision, cart_stiffness, cart_damping=None):
        """
        This function allows to move the KUKA EE towards the desired pose.
        The value of the EE pose is stored inside the RobotControlModule object, so a check
        is implemented to discriminate whether the pose is indeed valid.

        To guarantee that the robot joint configuration varies as smoothly as possible, we will use the
        ee_cartesian_jds mode: this means that the reference is a 6D Cartesian pose, but stiffness and damping
        must be expressed in joint space (7x1)
        """
        assert self.ee_desired_pose is not None, "Desired pose cannot be empty! Make sure of this before calling this function."

        # use ee_cartesian_ds mode to actually reach the cartesian pose we want
        self.reference_tracker.mode = 'ee_cartesian_jds'
        # retrieve the reference in a 6D format
        cartesian_position = self.ee_desired_pose[0:3, 3]
        euler_angles = np.flip(R.from_matrix(self.ee_desired_pose[0:3, 0:3]).as_euler('XYZ', degrees=False))    # the controller requires this angle sequence
        self.reference_tracker.reference = np.concatenate((cartesian_position, euler_angles))
        self.reference_tracker.time = time
        self.reference_tracker.rate = rate
        self.reference_tracker.precision = precision
        self.reference_tracker.stiffness = np.array([30, 30, 30, 30, 30, 30, 30])
        self.reference_tracker.damping = 2*np.sqrt(self.reference_tracker.stiffness)

        self.client.send_goal(self.reference_tracker)
        self.desired_pose_reached = self.client.wait_for_result()

        rospy.loginfo("Switch to Cartesian Mode")
        # use ee_cartesian_ds mode to actually reach the cartesian pose we want
        self.reference_tracker.mode = 'ee_cartesian_ds'

        # retrieve the reference in a 6D format
        cartesian_position = self.ee_desired_pose[0:3, 3]
        euler_angles = np.flip(R.from_matrix(self.ee_desired_pose[0:3, 0:3]).as_euler('XYZ', degrees=False))
        self.reference_tracker.reference = np.concatenate((cartesian_position, euler_angles))
        self.reference_tracker.time = 5
        self.reference_tracker.rate = rate
        self.reference_tracker.precision = precision
        self.reference_tracker.stiffness = cart_stiffness

        # set the damping
        if cart_damping is not None:
            self.reference_tracker.damping = cart_damping
        else:
            self.reference_tracker.damping = 2*np.sqrt(self.reference_tracker.stiffness)
        
        self.client.send_goal(self.reference_tracker)
        self.desired_pose_reached = self.client.wait_for_result()
        

    def trackReferenceStream(self, flag):
        """
        This function takes as inputs the boolean flag to toggle the tracking of the reference.
        """
        self.requesting_reference = flag

        if flag and not self.thread_therapy_status.is_alive():
            # start the thread the first time (only if it is not running already)
            self.thread_therapy_status.start()
        

    def requestUpdatedReference(self):
        """
        This function publishes on a default topic whether or not the robot is requesting an updated reference to track.
        This allows to perform all the start up procedures, and the optimal trajectory for the robot EE will be published
        only when the subject/experimenter are ready to start.
        """
        rate = rospy.Rate(10)        # setting the rate quite low so that the publishing happens not too often
        while not rospy.is_shutdown():
            self.pub_request_reference.publish(self.requesting_reference)
            rate.sleep()

    def cartesian_trajectory_done_callback(self, status, result):
        self.desired_pose_reached = result.success

    
    def joint_trajectory_done_callback(self, status, result):
        self.desired_pose_reached = result.success


if __name__ == "__main__":
    try:
        # check if we are running in simulation or not
        parser = argparse.ArgumentParser(description="Script that controls our Kuka robot")
        parser.add_argument("--simulation", required=True, type=str)
        args = parser.parse_args()
        simulation = args.simulation

        # define real-time factor if we are in simulation or not
        if simulation == 'true':
            rt_factor = 5
        else:
            rt_factor = 1

        rospy.loginfo("Running with simulation settings: ", simulation)
        rospy.loginfo("real time factor: ", rt_factor)

        # initialize ros node and set desired rate (imported above as a parameter)
        rospy.init_node("robot_control_module")
        ros_rate = rospy.Rate(loop_frequency)

        # flag that determines if the robotic therapy should keep going
        ongoing_therapy = True

        # initialize tkinter to set up a state machine in the code execution logic
        root = tk.Tk()
        root.title("Robot Interface (keep it selected)")

        # create the window that the user will have to keep selected to give their inputs
        window = tk.Canvas(root, width=500, height=200)
        window.pack()

        # Static caption
        caption_text = """Use the following keys to control the robot:
                        - 'a' to approach the starting pose for the therapy
                        - 's' to (re)start the therapy
                        - 'p' to pause the therapy
                        - 't' to run do some testing
                        - 'z' to set cartesian stiffness and damping to 0
                        - 'h' to send the robot to homing position
                        - 'c' to switch to the Cartesian controller
                        - 'j' to switch to the Joint controller
                        - 'q' to quit the therapy"""
        
        window.create_text(250, 100, text=caption_text, font=("Helvetica", 12), fill="black")


        # StringVar to store the pressed key
        pressed_key = tk.StringVar()

        def on_key_press(event):
            # update the StringVar with the pressed key
            pressed_key.set(event.char)

        # bind the key press event to the callback function
        root.bind("<Key>", on_key_press)

        # instantiate a RobotControlModule with the correct shared ros topics (coming from experiment_parameters.py)
        control_module = RobotControlModule(shared_ros_topics, experimental_params)

        rospy.loginfo("control module instantiated")

        # first make sure that we start from the homing position
        rospy.loginfo("Moving to homing position")
        control_module.reconf_client_joint.update_configuration({'stiffness':250})   # set stiffness for the controller
        goal = JointTrajectoryExecutionGoal()
        joint_pose_msg = Float64MultiArray()
        joint_pose_msg.data = np.zeros(7)       # retrieve the joint angles for the desired pose
        goal.joint_positions_goal = joint_pose_msg
        
        joint_vel_msg = Float64MultiArray()
        joint_vel_msg.data = rt_factor * 0.2 * np.ones(7)   # setting reference speet to 0.1 rad/s for all joints   
        goal.joint_velocities_goal = joint_vel_msg
        control_module.joint_action_client.send_goal(goal)
        control_module.joint_action_client.wait_for_result()

        rospy.loginfo("Homing position reached")

        while not rospy.is_shutdown() and ongoing_therapy:
            # state machine to allow for easier interface
            # it checks if there is a user input, and set parameters accordingly
            try:
                # wait for an event to occur
                event = root.wait_variable(pressed_key)

                # handle the event

                # 'a': approach initial pose
                if pressed_key.get() == "a":
                    if control_module.active_controller == 'JIC':
                        if control_module.getEEDesiredPose() is None:         # be sure that we know where to go
                            rospy.loginfo("waiting for initial pose to be known")
                            while control_module.getEEDesiredPose() is None:
                                ros_rate.sleep()

                        rospy.loginfo("Moving to initial pose")
                        # the retrieved goal is a homogenous matrix representing the desired pose for the robot's end-effector
                        homogenous_matrix_goal = control_module.getEEDesiredPose()
                        quat_orientation_goal = R.from_matrix(homogenous_matrix_goal[0:3, 0:3]).as_quat()   # x, y, z, w order

                        # build the goal to send to the controller (we assume that the controller is set to be a JointImpedanceController)
                        goal = JointTrajectoryExecutionGoal()
                        joint_pose_msg = Float64MultiArray()
                        joint_pose_msg.data = control_module.robot.ikine_LM(homogenous_matrix_goal, q0 = control_module.joint_states).q   # retrieve the joint angles for the desired pose
                        goal.joint_positions_goal = joint_pose_msg
                        
                        joint_vel_msg = Float64MultiArray()
                        joint_vel_msg.data = rt_factor * 0.2 * np.ones(7)  # setting reference speed to 0.2 rad/s for all joints   
                        goal.joint_velocities_goal = joint_vel_msg
                        control_module.joint_action_client.send_goal(goal, control_module.joint_trajectory_done_callback)
                        
                        # wait for the robot to reach the desired pose
                        control_module.joint_action_client.wait_for_result()

                        if control_module.desired_pose_reached:
                            rospy.sleep(0.1)    # wait a bit to make sure that the robot is stable

                            # switch to cartesian impedance controller
                            response = control_module.controller_manager(start_controllers=['/CartesianImpedanceController'],
                                                            stop_controllers=['/JointImpedanceController'], strictness=1,
                                                            start_asap=True, timeout=0.0)

                            if not response.ok:
                                rospy.logerr("Failed to switch to Cartesian controller")
                            else:
                                control_module.active_controller = 'CIC'

                            # add nullspace control on robot's elbow
                            control_module.reconf_client_cart.update_configuration({'cart_nullspace_control': True, 
                                                                            'elbow_ref_z':0.55, 
                                                                            'nullspace_stiffness_elbow_x':10,
                                                                            'nullspace_stiffness_elbow_y':10,
                                                                            'nullspace_stiffness_elbow_z':10})
                            
                            if experiment == 1:
                                # with the old controller I would add a nullspace joint controller also on the last links (to keep them close to 0)
                                # this is not implemented yet in the new controller
                                # TODO: test if we can maybe work around this by using only nullspace joint controller also above!
                                pass

                            # set the flag for indicating completion, and inform the user
                            control_module.initial_pose_reached = True
                            rospy.loginfo("Reached initial position. Starting to publish on topic %s (order data is [pe, se, ar])" % control_module.topic_shoulder_pose)
                            rospy.loginfo("-----------------------------------------------")
                            rospy.loginfo("Therapy can start")
                            rospy.loginfo("-----------------------------------------------")

                        else:
                            rospy.loginfo("Initial position could not be reached... Try again!")

                    elif control_module.active_controller == 'CIC':
                        rospy.loginfo("Robot is controlled by a CIC. Switching to a JIC...")
                        # switch to cartesian impedance controller
                        response = control_module.controller_manager(start_controllers=['/JointImpedanceController'],
                                                                     stop_controllers=['/CartesianImpedanceController'], strictness=1,
                                                                     start_asap=True, timeout=0.0)
                        if response.ok:
                            control_module.active_controller = 'JIC'
                        else:
                            rospy.logerr("Failed to switch to Joint controller")
                        
                        rospy.loginfo("Switched to JIC. Now you can try to move to the initial pose.")
                    else:
                        rospy.loginfo("Robot is controlled by an unknown controller. Cannot move to initial pose.")

                # 's' : start the therapy
                if pressed_key.get() == "s":
                    if control_module.initial_pose_reached:
                        # set experimentally tuned stiffness of np.array([550, 550, 550, 15, 15, 4])
                        control_module.reconf_client_cart.update_configuration({'separate_axis':False, 'translational_stiffness':550, 'rotational_stiffness':10})

                        rospy.loginfo("Start to follow optimal reference")
                        control_module.trackReferenceStream(True)
                    else:
                        rospy.loginfo("Robot is not in the initial pose yet. Doing nothing.")

                # 'p' : pause the therapy
                if pressed_key.get() == "p":
                    rospy.loginfo("Pause trajectory generation")
                    topic = shared_ros_topics['optimal_cartesian_ref_ee']
                    control_module.trackReferenceStream(False)

                # 't' : test something
                if pressed_key.get() == "t":
                    # Implement here the test we need
                    rospy.loginfo("No test implemented yet. Doing nothing")

                if pressed_key.get() == "h":

                    if control_module.active_controller == 'CIC':
                        rospy.loginfo("Robot is controlled by a CIC. Switching to a JIC...")
                        # switch to cartesian impedance controller
                        response = control_module.controller_manager(start_controllers=['/JointImpedanceController'],
                                                                     stop_controllers=['/CartesianImpedanceController'], strictness=1,
                                                                     start_asap=True, timeout=0.0)
                        if response.ok:
                            control_module.active_controller = 'JIC'
                        else:
                            rospy.logerr("Failed to switch to Joint controller")
                            
                    # send the robot to the home position
                    rospy.loginfo("Moving to homing position")
                    goal = JointTrajectoryExecutionGoal()
                    joint_pose_msg = Float64MultiArray()
                    joint_pose_msg.data = np.zeros(7)       # retrieve the joint angles for the desired pose
                    goal.joint_positions_goal = joint_pose_msg
                    
                    joint_vel_msg = Float64MultiArray()
                    joint_vel_msg.data = rt_factor * 0.2 * np.ones(7)   # setting reference speed to 0.2 rad/s for all joints   
                    goal.joint_velocities_goal = joint_vel_msg
                    control_module.joint_action_client.send_goal(goal)
                    control_module.joint_action_client.wait_for_result()

                # 'c' switch to cartesian impedance controller
                if pressed_key.get() == 'c':
                    if control_module.active_controller == 'JIC':
                            rospy.loginfo("switching controller to CIC")
                            response = control_module.controller_manager(start_controllers=['/CartesianImpedanceController'],
                                                                    stop_controllers=['/JointImpedanceController'], strictness=1,
                                                                    start_asap=True, timeout=0.0)

                            if not response.ok:
                                rospy.logerr("Failed to switch to cartesian controller")
                            else:
                                control_module.active_controller = 'CIC'
                    else:
                        rospy.loginfo("robot is already controlled by a CIC. Doing nothing...")
                    
                    # setting controller stiffness
                    control_module.reconf_client_cart.update_configuration({'separate_axis':False, 'translational_stiffness':400, 'rotational_stiffness':15})

                # 'j' switch to joint impedance controller
                if pressed_key.get() == 'j':
                    if control_module.active_controller == 'CIC':
                            rospy.loginfo("switching controller to JIC")
                            response = control_module.controller_manager(start_controllers=['/JointImpedanceController'],
                                                                    stop_controllers=['/CartesianImpedanceController'], strictness=1,
                                                                    start_asap=True, timeout=0.0)

                            if not response.ok:
                                rospy.logerr("Failed to switch to joint controller")
                            else:
                                control_module.active_controller = 'JIC'
                    else:
                        rospy.loginfo("robot is already controlled by a JIC. Doing nothing...")

                # 'z' set cartesian stiffness and damping to 0
                if pressed_key.get() == 'z':
                    rospy.loginfo("Zero stiffness/damping in 5 seconds")
                    time.sleep(2)
                    rospy.loginfo("3")
                    time.sleep(1)
                    rospy.loginfo("2")
                    time.sleep(1)
                    rospy.loginfo("1")
                    time.sleep(1)

                    # switch to zero stiffness and damping
                    control_module.reconf_client_cart.update_configuration({'set_new_reference_pose':True,
                                                                            'translational_stiffness':0, 
                                                                            'rotational_stiffness':0})
                    #confirm that everything went smoothly
                    rospy.loginfo("Free movement possible")

                # 'q': quit the execution, and freeze the robot to the current pose
                if pressed_key.get() == "q":
                    if control_module.active_controller == 'CIC':
                        rospy.loginfo("switching controller to JIC")
                        response = control_module.controller_manager(start_controllers=['/JointImpedanceController'],
                                                                stop_controllers=['/CartesianImpedanceController'], strictness=1,
                                                                start_asap=True, timeout=0.0)

                        if not response.ok:
                            rospy.logerr("Failed to switch to joint controller")
                        else:
                            control_module.active_controller = 'JIC'

                    rospy.loginfo("shutting down - freezing to current position")
                    control_module.requesting_reference = False

                    # adjust flag for termination
                    ongoing_therapy = False

            except tk.TclError:
                # root is destroyed (e.g., window closed)
                break
            # to allow ROS to execute other tasks if needed
            ros_rate.sleep()

    except rospy.ROSInterruptException:
        pass
