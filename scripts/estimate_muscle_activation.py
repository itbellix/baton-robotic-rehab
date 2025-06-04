import os
import numpy as np
import rospy
import time
from std_msgs.msg import Float32MultiArray

from RMRsolver import RMRsolver
import utilsObjectives as utilsObj
import warnings

# import the parameters for the experiment as defined in e xperiment_parameters.py
from experiment_parameters import *     # this contains the experimental_params and the shared_ros_topics

def estimate_muscle_activation(rmr_solver, experimental_params, shared_ros_topics):
    """
    This function leverages the RMR solver to estimate muscle activations of interest. It is called in a separate
    process to ensure responsive behaviour for the solver.
    """
    # initialize new node
    rospy.init_node('muscle_activity_estimator', anonymous=True)

    # initialize variables
    state = {
        'position': None,
        'velocity': None,
        'acceleration': None,
        'ft_reading': np.zeros(6),
        'ft_sensor_is_functional': False
    }

    def _shoulder_state_rmr_cb(msg):
        state_values_current = np.array(msg.data[0:6])
        state_dot_current = np.array([msg.data[i] for i in [1, 6, 3, 7, 5, 8]])
        state['position'] = state_values_current[0::2]
        state['velocity'] = state_values_current[1::2]
        state['acceleration'] = state_dot_current[1::2]

    def _ft_data_rmr_cb(msg):
        state['ft_sensor_is_functional'] = True
        state['ft_reading'] = np.array(msg.data[0:-2])   # last two elements are temperature and timestamp

    rospy.Subscriber(shared_ros_topics['estimated_shoulder_pose'], Float32MultiArray, _shoulder_state_rmr_cb)
    rospy.Subscriber(shared_ros_topics['ft_sensor_data'], Float32MultiArray, _ft_data_rmr_cb)

    pub_activation = rospy.Publisher(shared_ros_topics['muscle_activation'], Float32MultiArray, queue_size=10)
    pub_compensated_wrench = rospy.Publisher(shared_ros_topics['compensated_wrench'], Float32MultiArray, queue_size=10)

    rate = rospy.Rate(experimental_params['ros_rate'])

    while not rospy.is_shutdown():
        if state['position'] is not None:
            # retrieve the interaction wrenches from the force-torque sensor
            if state['ft_sensor_is_functional']:
                # get the interaction wrench from the force-torque sensor
                # note that we set fz = mx = my = 0 as only the other values affect muscle
                # activation when the arm is locked in the brace, and the GH center does not move.
                interaction_forces = experimental_params['ulna_R_sensor'].as_matrix() @ state['ft_reading'][0:3]
                interaction_torques = experimental_params['ulna_R_sensor'].as_matrix() @ state['ft_reading'][3:6]
                interaction_wrench = - np.hstack((interaction_forces, interaction_torques)) # note the minus sign, as the opposite of what the
                                                                                            # sensor measures is felt by the human body

            else:
                interaction_wrench = np.zeros(6)

            # estimate the muscle activation level for all the muscles in the model
            current_activation, _, info = rmr_solver.solve(time = time.time(), 
                                                            position = state['position'], 
                                                            speed = state['velocity'], 
                                                            acceleration = state['acceleration'], 
                                                            values_prescribed_forces = interaction_wrench)
            
            # publish the activation level (for debugging and logging)
            message = Float32MultiArray()
            message.data = current_activation
            pub_activation.publish(message)

            # publish the interaction wrenches after gravity compensation (for debugging and logging)
            message = Float32MultiArray()
            message.data = interaction_wrench
            pub_compensated_wrench.publish(message)

        # sleep for a while
        rate.sleep()


if __name__ == '__main__':

    # we will suppress runtime warnings to keep terminal output clean
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
    path_to_model = os.path.join(path_to_repo, 'Musculoskeletal Models')    # getting path to the OpenSim models
    model_name_osim = 'right_arm_GH_full_scaled_preservingMass_muscles.osim'

    # try to import the OpenSim module
    try:
        import opensim
    except ImportError:
        print("The module OpenSim could not be imported. Troubleshooting for why this is happening:")
        print("option 1: you have not built OpenSim from source on your machine\n\t  (find instructions on how to do so at https://github.com/opensim-org/opensim-core/wiki/Build-Instructions)")
        print("option 2: you have not activated the correct (conda) environment")
        with_opensim = False
        rmr_solver = None
        print("Proceeding: no muscle activity will be estimated...")
        
    else:
        with_opensim = True
        opensim_model = opensim.Model(os.path.join(path_to_model, model_name_osim))

        # select the ulna reference frame as the frame to which the interaction forces between human and robot are applied
        ulna_body = opensim_model.updBodySet().get('ulna')
        prescribed_force_ulna = opensim.PrescribedForce("ulna_force", ulna_body)
        prescribed_force_ulna.setPointIsInGlobalFrame(False)
        prescribed_force_ulna.setForceIsInGlobalFrame(False)
        prescribed_force_ulna.setPointFunctions(opensim.Constant(0.0), opensim.Constant(0.0), opensim.Constant(0.0))     # point of application of the force in body reference frame
        prescribed_force_ulna.setForceFunctions(opensim.Constant(0.0), opensim.Constant(0.0), opensim.Constant(0.0))
        prescribed_force_ulna.setTorqueFunctions(opensim.Constant(0.0), opensim.Constant(0.0), opensim.Constant(0.0))

        opensim_model.addForce(prescribed_force_ulna)

        # instantiate the RMR solver object
        print("OpenSim found, RMR solver will be used")
        weights = np.concatenate((np.ones(22), 10*np.ones(3)))
        objective = utilsObj.ActSquared(weights)
        rmr_solver = RMRsolver(opensim_model, 
                                constrainActDyn=True, 
                                constrainGHjoint=True, 
                                actuatorReserveBounds=[-1e6, 1e6], 
                                prescribedForceIndex = [opensim_model.getForceSet().getIndex(prescribed_force_ulna)])
        rmr_solver.setObjective(objective)

        while not rospy.is_shutdown():
            estimate_muscle_activation(rmr_solver=rmr_solver, 
                                       experimental_params=experimental_params,
                                       shared_ros_topics=shared_ros_topics)
