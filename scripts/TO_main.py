"""
This script contains the Trajectory Optimization module to perform musculoskeletal-aware trajectory planning
for rotator cuff rehabilitation, based on an OpenSim shoulder model.
Here, we employ the thoracoscapular shoulder model (available at https://simtk.org/projects/scapulothoracic), already
scaled to our subject. Further, the models used in the script capture specifically the mobility of the glenohumeral
joint in the shoulder, with the elbow being locked at 90 degrees to mimic our experimental setup.

To run this code, OpenSimAD is required. This can be done following the instructions on the official repository,
available at https://github.com/antoinefalisse/opensimAD
Once the OpenSimAD virtual environment is setup, proceed as follows:
1. open a terminal, source your ROS distribution (tested with Noetic) and iiwa-ros (available at https://gitlab.tudelft.nl/kuka-iiwa-7-cor-lab/iiwa_ros)
      - launch the ROS master and (optionally) the Gazeo simulation by running the controller.launch from Code/launch/ in this repository
2. open another terminal, source your ROS distribution and activate the OpenSimAD virtual environment
      - run this code ($python TO_main --simulation=true)
3. open a third terminal, source your ROS distribution and iiwa-ros
      - run $python robot_control.py --simulation=true

... and start generating musculoskeletal-aware rehabilitation trajectories!
"""

import os
import casadi as ca
import numpy as np
import rospy
import time
import matplotlib.pyplot as plt
import utilities_TO as utils_TO
import utilities_casadi as utils_ca
from scipy.spatial.transform import Rotation as R
import pickle

from botasensone import BotaSerialSensor
from RMRsolver import RMRsolver
import utilsObjectives as utilsObj
import warnings

# import parser
import argparse

if __name__ == '__main__':

    # we will suppress runtime warnings to keep terminal output clean
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    try:
        # check if we are running in simulation or not
        parser = argparse.ArgumentParser(description="Script that runs the Trajectory Optimization")
        parser.add_argument("--simulation", required=True, type=str)
        args = parser.parse_args()
        simulation = args.simulation

        # define the required paths
        code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
        path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
        path_to_model = os.path.join(path_to_repo, 'Musculoskeletal Models')    # getting path to the OpenSim models

        # human musculoskeletal models
        # Note: model_name_AD is the differentiable version of model_name_osim, obtained thanks to OpenSimAD.
        # They both capture the biometrics of our test subject (mass, inertia and lengths of the segments).
        model_name_osim = 'right_arm_GH_full_scaled_preservingMass_muscles.osim'  # OpenSim model (should be in the path provided)
        model_name_AD = 'right_arm_GH_full_scaled_preservingMass.casadi'  # CasADi function capturing model dynamics

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
                                   prescribedForceIndex = [opensim_model.getForceSet().getIndex(prescribed_force_ulna)],
                                   visualize = True)
            rmr_solver.setObjective(objective)

            # debug
            exp_wrench = np.zeros((6, 22))
            exp_wrench[2,0:] = 70 * np.linspace(-1, 1, 22)
            # exp_wrench[3,0:] = 5 * np.linspace(-1, 1, 22)

            position_sh = np.deg2rad(np.array([35, 89, 13]))
            velocity_sh = np.zeros((3, 1))
            acceleration_sh = np.zeros((3,1))

            current_activation = np.zeros((25, 22))

            for i in range(22):
                if i==21:
                    aux = 0
                current_activation[:,i], _, _ = rmr_solver.solve(time = time.time(), 
                                                                position = position_sh.squeeze(), 
                                                                speed = velocity_sh.squeeze(), 
                                                                acceleration = acceleration_sh.squeeze(), 
                                                                values_prescribed_forces = exp_wrench[:,i])

        ## PARAMETERS -----------------------------------------------------------------------------------------------
        # import the parameters for the experiment as defined in e xperiment_parameters.py
        from experiment_parameters import *     # this contains the experimental_params and the shared_ros_topics

        # are we debugging or not?
        debug = False
        debug_solver_warmstart = False

        # what will define our system dynamics?
        use_casadi_function = True          # discriminates whether to use pre-saved CasADi function (through OpenSimAD), 
                                            # or numerical CasADi+OpenSim callback

        # choose collocation scheme for approximating the system dynamics
        collocation_scheme = 'legendre'         # collocation points
        polynomial_order = 3                    # order of polynomial basis

        # constraints on generalized control torques (for plane of elevation and shoulder elevation DoFs)
        # note that the latter will need to oppose gravity too
        u_max = 10       # Nm
        u_min = -10     # Nm

        # choose solver and set its options
        solver = 'ipopt'        # available solvers depend on CasADi interfaces

        # quick check to determine if linear solver ma27 from HSL is available. Otherwise, we use MUMPS
        # Define a simple NLP problem for this use
        print("checking which linear solver is available...")
        chosen_solver = "ma27" if utils_TO.is_hsl_present() else "mumps"

        if chosen_solver == "ma27":
            print("ma27 will be used")
            opts = {'ipopt.print_level': 5,         # options for the solver (check CasADi/solver docs for changing these)
                    'print_time': 0,
                    'ipopt.tol': 1e-3,
                    'error_on_fail':1,              # to guarantee transparency if solver fails
                    'ipopt.linear_solver':'ma27'}
        elif chosen_solver == "mumps":
            print("mumps will be used")
            opts = {'ipopt.print_level': 5,         # options for the solver (check CasADi/solver docs for changing these)
                    'print_time': 0,
                    'ipopt.tol': 1e-3,
                    'error_on_fail':1,              # to guarantee transparency if solver fails
                    'ipopt.linear_solver':'mumps'}
        else:
            rospy.ERROR("No linear solver available. Please install HSL for ma27 or MUMPS for mumps.")

        if use_casadi_function:
            opts['expand'] = 1  # allows to speed up code (only for OpenSimAD-generated function)
        else:
            opts['ipopt.hessian_approximation'] = 'limited-memory'      # for numerical callback, to avoid expensive hessian evaluation

        if experiment == 2 or experiment == 3:
            opts['ipopt.max_iter'] = 100    # reduce the number of iterations, to restart the optimization from new state
        # ----------------------------------------------------------------------------------------------------------------
        # Declare model variables
        # The model represents a right arm, whose only degrees of freedom are the ones of the glenohumeral (GH) joint,
        # based on the thoracoscapular shoulder model from https://doi.org/10.3389/fnbot.2019.00090
        # The coordinates are plane of elevation, shoulder elevation and axial rotation (theta, psi and phi, for brevity).
        # However, the axial rotation (phi) is not controlled, as we leave it free for the patient to adjust it.
        dim_x = 6
        dim_u = 2

        x = ca.MX.sym('x', dim_x)   # state vector: [theta, theta_dot, psi, psi_dot, phi, phi_dot], in rad or rad/s

        u = ca.MX.sym('u', dim_u)   # control vector: [tau_theta, tau_psi], in Nm (along 2 coordinates of GH joint, phi is not controlled)       

        # instantiate NLP problem
        nlps_module = utils_TO.nlps_module()

        # define how the dynamics of the model will be provided to the optimizer
        if use_casadi_function: 
            # using CasADi function (obtained through OpenSimAD)
            nlps_module.setSystemDynamics(ca.Function.load(os.path.join(path_to_model, 
                                                                        model_name_AD)))
        else:
            if not with_opensim:
                RuntimeError("If the dynamics is provided through a callback, the OpenSim module is required.")

            # setting the callback that will provide the dynamics
            nlps_module.setSystemDynamics(utils_ca.MyOsimCallback('sys_dynamics', 
                                                                  opensim_model, 
                                                                  {"enable_fd":True}))

        # initialize the NLP problem with its parameters
        nlps_module.setTimeHorizonAndDiscretization(N=N, T=T)
        nlps_module.populateCollocationMatrices(order_polynomials= polynomial_order, collocation_type= collocation_scheme)

        # provide states and controls to the NLP instance
        nlps_module.initializeStateVariables(x = x, names = ['theta', 'theta_dot', 'psi', 'psi_dot', 'phi', 'phi_dot'])
        nlps_module.initializeControlVariables(u = u, names= ['tau_theta', 'tau_psi'])

        # set the goal to be reached, and initial condition we are starting from
        if len(x_goal.shape) == 1:
            nlps_module.setGoal(goal = x_goal)
        else:
            nlps_module.setGoal(goal = x_goal[0,:])
        nlps_module.setInitialState(x_0 = x_0)

        # instantiate the force/torque sensor object
        print("Connecting force-torque sensor...")
        sensor = BotaSerialSensor(experimental_params['ft_sensor_port'], n_readings_calib=1000)

        # instantiate trajectory optimization module, given the NLP problem, 
        # the shared ros topics (defined in experiment_parameters.py), and setting debugging options
        to_module = utils_TO.TO_module(nlps = nlps_module, 
                                    shared_ros_topics=shared_ros_topics, 
                                    rate = loop_frequency, 
                                    with_opensim = with_opensim,
                                    simulation = simulation,
                                    speed_estimate = speed_estimate,
                                    ft_sensor=sensor,
                                    rmr_solver = rmr_solver)

        # set the strainmap to operate onto, extracting the information from a file
        if file_strainmaps is not None:
            with open(file_strainmaps, 'rb') as file:
                strainmaps_dict = pickle.load(file)
        else:
            strainmaps_dict = None

        to_module.setStrainMapsParamsDict(strainmaps_dict)

        if experiment == 2:
            # set also information about the activation information captured by the strainmaps
            to_module.setActivationBounds(max_activation=max_activation, min_activation=min_activation, delta_activation=delta_activation)

        # define the base cost function (drive the state towards the goal and minimize control action)
        # Note that, in the formalization of the actual NLP, other terms will be added
        init_state = ca.MX.sym('x_0', dim_x)
        desired_state = ca.MX.sym('x_d', dim_x)
        L = gamma_goal*((x[0]-desired_state[0, 0])**2 + (x[2]-desired_state[2, 0])**2)/ca.sumsqr(init_state[[0,2]]-desired_state[[0,2]])

        cost_function = ca.Function('cost_function', [x, u, init_state, desired_state], [L])
        to_module.nlps_module.setCostFunction(cost_function = cost_function)

        # other terms of the cost function are easier to treat in the NLP structure directly
        to_module.nlps_module.setCostWeights(goal = gamma_goal,
                                            strain = gamma_strain,
                                            velocities = gamma_velocities,
                                            acceleration = gamma_acceleration)

        # enforce the terminal constraint
        to_module.nlps_module.enforceFinalConstraints(on_position = constrain_final_position, on_velocity = constrain_final_velocity)

        # Publish the initial position of the KUKA end-effector, according to the initial shoulder state
        # This code is blocking until an acknowledgement is received, indicating that the initial pose has been successfully
        # received by the RobotControlModule
        to_module.publishInitialPoseAsCartRef(shoulder_pose_ref = x_0[0::2], 
                                            position_gh_in_base = experimental_params['p_gh_in_base'], 
                                            base_R_sh = experimental_params['base_R_shoulder'], 
                                            dist_gh_elbow = experimental_params['d_gh_ee_in_shoulder'])

        # Wait until the robot has reached the required position, and proceed only when the current shoulder pose is published
        to_module.waitForShoulderState()

        # set up the NLP
        if use_casadi_function:
            to_module.nlps_module.formulateNLP_functionDynamics(constraint_list = {'u_max':u_max, 'u_min':u_min})
        else:
            to_module.nlps_module.formulateNLP_callbackDynamics(constraint_list = {'u_max':u_max, 'u_min':u_min})

        to_module.nlps_module.setSolverOptions(solver = solver, opts = opts)

        # create a function that solves the NLP given numerical values for its parameters
        to_module.createMPCfunctionWithoutInitialGuesses()

        if debug_solver_warmstart:
            to_module.nlps_module.solveNLPOnce()
            to_module.createMPCfunctionInitialGuesses()

            # use the function created above to solve the same problem a couple of times
            num_iterations = 5

            time_iters_noInitGuess = np.zeros((num_iterations, 1))
            time_iters_initGuess = np.zeros((num_iterations, 1))

            x_opt = np.zeros(to_module.nlps_module.getSizePrimalVars()[0])
            u_opt = np.zeros(to_module.nlps_module.getSizePrimalVars()[1])

            lam_g_prev = np.zeros(to_module.nlps_module.getSizeDualVars())

            params_g1 = to_module.nlps_module.all_params_gaussians[0:6]
            params_g2 = to_module.nlps_module.all_params_gaussians[6:12]
            params_g3 = to_module.nlps_module.all_params_gaussians[12:18]

            for iter in range(num_iterations):
                    start_iter = time.time()
                    u_opt, x_opt, lam_g_prev, j_opt, strain_opt = to_module.MPC_iter(x_0, x_goal[0,:], params_g1, params_g2, params_g3)
                    time_iters_noInitGuess[iter] = time.time()-start_iter

                    u_opt = u_opt + np.random.normal(0,0.2, 20)
                    x_opt = x_opt + np.random.normal(0,0.2, 66)
                    lam_g_prev = lam_g_prev + np.random.normal(0, 0.1, lam_g_prev.shape[0])

                    start_iter = time.time()
                    u_opt, x_opt, lam_g_prev = to_module.MPC_iter_initGuess(x_0, x_goal[0,:], params_g1, params_g2, params_g3, u_opt, x_opt, lam_g_prev)
                    time_iters_initGuess[iter] = time.time()-start_iter
                    print(iter, '\n')

            plt.plot(time_iters_noInitGuess, marker='o', label='no initGuess')
            plt.plot(time_iters_initGuess, marker='*', label='with initGues')
            plt.xlabel('Iteration')
            plt.ylabel('Execution Time (s)')
            plt.legend()

        # wait until the robot is requesting the optimized reference
        print("Waiting for the robot to request optimized reference")
        while not to_module.keepOptimizing():
            # to_module.nlps_module.strain_visualizer.update_strain_map(to_module.nlps_module.all_params_gaussians)
            rospy.sleep(0.1)

        # countdown for the user to be ready
        print("Starting to provide optimal reference")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)

        # start moving towards the first goal
        # then keep selecting new goals if the user has provided more
        n_goals = x_goal.shape[0]
        goal_index = 0
        delta_act = 0.005
        current_activation = 0.0
        count = 0

        to_module.time_begin_optimizing = time.time()

        # start to provide the optimal reference as long as the robot is requesting it
        # we do so in a way that allows temporarily pausing the therapy
        # Note: the structure of the following loop accounts for running all of the experiments that are reported in our
        # paper. Please check experiment_parameters.py to get an idea of the various experiments performed.
        while not rospy.is_shutdown():
            if to_module.keepOptimizing() and goal_index<n_goals:
                # determine the current_goal
                if len(x_goal.shape)>1:
                    current_goal = x_goal[goal_index, :]
                else:
                    current_goal = x_goal

                if perform_BATON:
                    # optimize the trajectory towards the given goal
                    to_module.nlps_module.setGoal(goal = current_goal)
                    to_module.optimize_trajectory(current_goal, delay = 0.1)
                    previous_request = True         # save the status of the robot request now

                    if experiment == 1:
                        rospy.sleep(6)
                        goal_index += 1
                    
                    if experiment == 2:

                        if target_tendon == "SSPA":

                            if goal_index==2:
                                if current_activation+delta_act>0.25 or current_activation+delta_act<0.00:
                                    delta_act = - delta_act
                                current_activation += delta_act
                                to_module.setActivationLevel(current_activation)

                            if goal_index>2:
                                # to_module.setActivationLevel(0.99)        # this is for having the real robot work fine 
                                to_module.setActivationLevel(0.25)           # this is for having the simulation work on the real strainmap
                            
                            if to_module.reachedPose(current_goal, tolerance = 0.05):
                                goal_index += 1
                                # time.sleep(3)

                        if target_tendon == "SSPA_sim":
                            count = count + 1
                            # uncomment to impose an activation ramp during simulation
                            # if current_activation+delta_act>0.25 or current_activation+delta_act<0.00:
                            #     delta_act = - delta_act
                            # current_activation += delta_act
                            # to_module.setActivationLevel(current_activation)

                            # uncomment to impose an activation step during simulation
                            if count <30:
                                to_module.setActivationLevel(0.0)
                            else:
                                to_module.setActivationLevel(0.25)
                            

                        if target_tendon == "custom_1":
                            # this generates extreme variations in the strain map considered.
                            # The activation levels set here are fictitious, they are used in utilities_TO.py (specifically in _shoulder_pose_cb())
                            # to precisely generate different strain landscapes.
                            if count < 20:
                                to_module.setActivationLevel(0.90)
                            elif count >= 20 and count < 60:
                                to_module.setActivationLevel(0.91)
                            elif count >= 60:
                                to_module.setActivationLevel(0.92)
                                
                            count += 1

                elif perform_A_star:
                    # optimize the trajectory towards the given goal
                    to_module.optimize_trajectory_astar(maze, current_goal, strainmap_list_astar)
                    previous_request = True         # save the status of the robot request now

                    rospy.sleep(6)
                    goal_index += 1

            
            # if the user wants to interrupt the therapy, or if the last goal position has been reached,
            # we stop the optimization and freeze the robot to its current position.
            # Also, we print some statistics regarding the execution of the algorithm until now.
            if not to_module.keepOptimizing() or goal_index == n_goals:
                # overwrite the optimal trajectory with the current state
                if experiment == 1:
                    to_module.setOptimalReferenceToCurrentOptimalPose()   # to decrease bumps (as trajectory will be very smooth anyway)

                if previous_request:
                    # print the stats to screen
                    to_module.getStats(True)

                previous_request = False        # save the status of the robot request now
                goal_index += 1                 # using this as an extra trigger

            to_module.ros_rate.sleep()

    except rospy.ROSInterruptException:
        pass
