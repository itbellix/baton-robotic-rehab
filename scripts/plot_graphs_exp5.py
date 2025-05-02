"""
Script to analyze the data collected in a rosbag during the experiment regarding the Passive Human Case.
The plots produced here are used in Fig.5 and Fig.6 of the paper.
"""

import os
import pickle
from spatialmath import SO3
import numpy as np
import rosbag
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

num_params = 6
def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    '''
    Function used for the 2D interpolation
    '''
    return amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2)+(y-y0)**2/(2*sigma_y**2)))+offset

def generate_approximated_strainmap(file_strainmaps, ar_value):
    """
    Function to generate the approximated strainmap to consider, given a file containing the
    corresponding dictionary, and the axial rotation value with which to slice it.
    """

    # set the strainmap to operate onto, extracting the information from a file
    with open(file_strainmaps, 'rb') as file:
        strainmaps_dict = pickle.load(file)

    # strainmap span
    max_se = 144
    min_se = 0

    max_pe = 160
    min_pe = -20

    min_ar = -90
    max_ar = 102

    step_from_model = 4
    step = 0.05

    # create the grids that will be used for interpolation (and visualization)
    pe_datapoints = np.array(np.arange(min_pe, max_pe, step))
    se_datapoints = np.array(np.arange(min_se, max_se, step))
    ar_datapoints = np.array(np.arange(min_ar, max_ar, step_from_model))

    X,Y = np.meshgrid(pe_datapoints, se_datapoints, indexing='ij')
    X_norm = X/max_pe
    Y_norm = Y/max_se

    index_strainmap_current = np.argmin(np.abs(ar_datapoints - np.rad2deg(ar_value)))   # index of the strainmap
    params_gaussians = strainmaps_dict['all_params_gaussians'][index_strainmap_current]

    fit = np.zeros((pe_datapoints.shape[0], se_datapoints.shape[0]))
    for i in range(len(params_gaussians)//num_params):
        fit += gaussian_2d(X_norm, Y_norm, *params_gaussians[i*num_params:i*num_params+num_params])

    return fit



def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_5')
    # bag_file_name = '2024-03-26-12-42-34_Exp1_noHuman.bag'
    # bag_file_name = '2024-03-26-12-46-15_Exp1_withHuman.bag'
    # bag_file_name = 'experiment_1/exp1_good_try_withHuman.bag'
    bag_file_name = 'exp5_2.bag'

    # load the strainmap dictionary used in the experiment
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models','Strain Maps','Passive','differentiable_strainmaps_allTendons.pkl')
    
    # instantiate variables (they will be Mx4 matrices, where M is variable -number of data- and the last column is the timestamp)
    estimated_shoulder_state = None
    xyz_cmd = None
    xyz_curr = None
    z_uncompensated = None
    angvec_cmd = None
    angvec_curr = None

    with rosbag.Bag(os.path.join(path_to_bag, bag_file_name), 'r') as bag:
        # extracting estimated shoulder poses
        print('Extracting estimated shoulder poses')
        for _, msg, time_msg in bag.read_messages(topics=['/estimated_shoulder_pose']):
            if estimated_shoulder_state is None:
                estimated_shoulder_state = [msg.data + (time_msg.to_time(),)]
            else:
                estimated_shoulder_state = np.vstack((estimated_shoulder_state, [msg.data + (time_msg.to_time(),)]))

        # extracting commanded ee cartesian poses
        print('Extracting commanded ee cartesian poses')
        for _, msg, time_msg in bag.read_messages(topics=['/CartesianImpedanceController/reference_cartesian_pose']):
            if xyz_cmd is None:
                timestamps_cmd = time_msg.to_time()
                position_reference = np.array([msg.pose.position.x,msg.pose.position.y, msg.pose.position.z])
                quat_reference = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                xyz_cmd = np.hstack((position_reference, timestamps_cmd))
                angle, vector = SO3(R.from_quat(quat_reference, scalar_first=False).as_matrix()).angvec(unit='deg')
                angvec_cmd = np.hstack((angle, vector, timestamps_cmd))
            else:
                timestamps_cmd = time_msg.to_time()
                position_reference = np.array([msg.pose.position.x,msg.pose.position.y, msg.pose.position.z])
                quat_reference = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                xyz_cmd = np.vstack((xyz_cmd, np.hstack((position_reference, timestamps_cmd))))
                angle, vector = SO3(R.from_quat(quat_reference, scalar_first=False).as_matrix()).angvec(unit='deg')
                angvec_cmd = np.vstack((angvec_cmd, np.hstack((angle, vector, timestamps_cmd))))

        print('Extracting not adjusted ee Z cartesian value')
        for _, msg, time_msg in bag.read_messages(topics=['/uncompensated_z_ref']):
            if z_uncompensated is None:
                timestamps_z = time_msg.to_time()
                z_uncompensated = np.hstack((msg.data[0], timestamps_z))
            else:
                timestamps_z = time_msg.to_time()
                z_uncompensated = np.vstack((z_uncompensated, np.hstack((msg.data[0], timestamps_z))))

        # extracting actual ee cartesian poses
        print('Extracting actual ee cartesian poses')
        for _, msg, time_msg in bag.read_messages(topics=['/CartesianImpedanceController/cartesian_pose']):
            time_curr = time_msg.to_time()
            if time_curr >= xyz_cmd[0, -1]:
                if xyz_curr is None:
                    timestamps_curr = time_msg.to_time()
                    position_curr = np.array([msg.pose.position.x,msg.pose.position.y, msg.pose.position.z])
                    quat_curr = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                    xyz_curr = np.hstack((position_curr, timestamps_curr))
                    angle, vector = SO3(R.from_quat(quat_curr, scalar_first=False).as_matrix()).angvec(unit='deg')
                    angvec_curr = np.hstack((angle, vector, timestamps_curr))
                else:
                    timestamps_curr = time_msg.to_time()
                    position_curr = np.array([msg.pose.position.x,msg.pose.position.y, msg.pose.position.z])
                    quat_curr = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                    xyz_curr = np.vstack((xyz_curr, np.hstack((position_curr, timestamps_curr))))
                    angle, vector = SO3(R.from_quat(quat_curr, scalar_first=False).as_matrix()).angvec(unit='deg')
                    angvec_curr = np.vstack((angvec_curr, np.hstack((angle, vector, timestamps_curr))))

        print('Extracting optimization outputs')
        # assume that A* output is resampled to produce 50 points per trajectory
        n_interp = 250
        optimal_trajectory = None
        for _, msg, time_msg in bag.read_messages(topics=['/optimization_output']):
            data = np.array(msg.data).reshape((6, n_interp))
            time_curr = time_msg.to_time()
            if optimal_trajectory is None:
                optimal_trajectory = np.hstack((data, time_curr * np.ones((6,1))))
            else:
                optimal_trajectory = np.vstack((optimal_trajectory, np.hstack((data, time_curr * np.ones((6,1))))))

        print('Extracting forces')
        interaction_force_magnitude = None
        for _, msg, time_msg in bag.read_messages(topics=['/ft_sensor_data']):
            time_curr = time_msg.to_time()
            if interaction_force_magnitude is None:
                interaction_force_magnitude = np.hstack((np.linalg.norm(msg.data[0:3]), time_curr))
            else:
                interaction_force_magnitude = np.vstack((interaction_force_magnitude, np.hstack((np.linalg.norm(msg.data[0:3]), time_curr))))

    num_optim = int(optimal_trajectory.shape[0]/6)

    # Now, let's filter the data to retain only the interesting part of the experiment
    # (i.e., when the subject is wearing the brace properly and the robot is moving)
    init_time = xyz_curr[int(xyz_curr.shape[0]/100*40), -1]         # identify initial timestep
    end_time = xyz_curr[int(xyz_curr.shape[0]/100*90), -1] - init_time # identify final timestep

    estimated_shoulder_state[:,-1] = estimated_shoulder_state[:,-1] - init_time   # center time values starting at initial time
    xyz_curr[:,-1] = xyz_curr[:,-1] - init_time
    xyz_cmd[:,-1] = xyz_cmd[:,-1] - init_time
    if z_uncompensated is not None:
        z_uncompensated[:,-1] = z_uncompensated[:,-1] - init_time
    angvec_curr[:,-1] = angvec_curr[:,-1] - init_time
    angvec_cmd[:,-1] = angvec_cmd[:,-1] - init_time
    optimal_trajectory[:,-1] = optimal_trajectory[:,-1] - init_time

    if interaction_force_magnitude is not None:
        interaction_force_magnitude[:,-1] = interaction_force_magnitude[:,-1] - init_time

    estimated_shoulder_state = estimated_shoulder_state[(estimated_shoulder_state[:,-1]>0) & (estimated_shoulder_state[:,-1]<end_time)]
    xyz_curr = xyz_curr[(xyz_curr[:,-1]>0) & (xyz_curr[:,-1]<end_time)]    # retain data after initial time
    xyz_cmd = xyz_cmd[(xyz_cmd[:,-1]>0) & (xyz_cmd[:,-1]<end_time)]
    if z_uncompensated is not None:
        z_uncompensated = z_uncompensated[(z_uncompensated[:,-1]>0) & (z_uncompensated[:,-1]<end_time)]
    angvec_curr = angvec_curr[(angvec_curr[:,-1]>0) & (angvec_curr[:,-1]<end_time)]
    angvec_cmd = angvec_cmd[(angvec_cmd[:,-1]>0) & (angvec_cmd[:,-1]<end_time)]

    if interaction_force_magnitude is not None:
        interaction_force_magnitude = interaction_force_magnitude[(interaction_force_magnitude[:,-1]>0) & (interaction_force_magnitude[:,-1]<end_time)]
    
    strainmap = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[100, 4])

    # visualize 2D trajectory on strainmap
    fig = plt.figure()
    ax = fig.add_subplot()
    cb =ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=0, vmax=strainmap.max())
    fig.colorbar(cb, ax=ax, label = 'Strain [%]')
    ax.plot(np.rad2deg(estimated_shoulder_state[:-1, 0]), np.rad2deg(estimated_shoulder_state[:-1, 2]))
    ax.scatter(np.array([50]), np.array([100]), label = 'start')
    ax.scatter(np.array([100]), np.array([100]), label = 'goal1')
    ax.scatter(np.array([50]), np.array([70]), label = 'goal2')
    ax.scatter(np.array([60]), np.array([110]), label = 'goal3', c='black')
    # plot also the optimal trajectories
    ax.plot(np.rad2deg(optimal_trajectory[0,:-1]), np.rad2deg(optimal_trajectory[2,:-1]))
    ax.plot(np.rad2deg(optimal_trajectory[6,:-1]), np.rad2deg(optimal_trajectory[8,:-1]))
    ax.plot(np.rad2deg(optimal_trajectory[12,:-1]), np.rad2deg(optimal_trajectory[14,:-1]))
    ax.set_xlabel("Plane of elevation [deg]")
    ax.set_ylabel("Shoulder elevation [deg]")
    ax.legend()
    ax.set_title("EXP1: Trajectory on strainmap (with human)")

    # visualize 2D trajectory on strainmap
    fig = plt.figure()
    ax = fig.add_subplot()
    cb =ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=1.5, vmax=3.2)
    fig.colorbar(cb, ax=ax, label = 'Strain [%]')
    ax.scatter(np.array([50]), np.array([100]), label = 'start', edgecolors='black', s=50, zorder=1)
    ax.scatter(np.array([100]), np.array([100]), label = 'goal1', edgecolors='black', s=50, zorder=1)
    ax.scatter(np.array([60]), np.array([60]), label = 'goal2', edgecolors='black', s=50, zorder=1)
    ax.scatter(np.array([60]), np.array([110]), label = 'goal3', edgecolors='black', c='pink', s=50, zorder=1)
    ax.plot(np.rad2deg(estimated_shoulder_state[:-1, 0]), np.rad2deg(estimated_shoulder_state[:-1, 2]), linewidth=2, zorder=0)
    ax.axis('equal')
    ax.set_xlim(30, 120)
    ax.set_ylim(60, 120)
    ax.set_xlabel("Plane of elevation [deg]")
    ax.set_ylabel("Shoulder elevation [deg]")
    ax.legend()
    ax.set_title("EXP1: Trajectory on strainmap (with human)")

    # # visualize strain along the optimal path, and along the quickest trajectory
    # # Note, in the following there is quite a lot of hardcoded things, to select the correct data for this particular run of the experiment
    # This is not the best way to visualize these things I think
    min_pe = -20
    min_se = 0
    step = 0.05
    num_samples = 51
    time_beginning_1 = optimal_trajectory[0,-1]
    index_beginning_1 = np.where(estimated_shoulder_state[:,-1]-time_beginning_1>-4e-3)[0][0]
    position_beginning_1 = np.rad2deg(estimated_shoulder_state[index_beginning_1, [0, 2]])

    time_beginning_2 = optimal_trajectory[6,-1]
    index_beginning_2 = np.where(abs(estimated_shoulder_state[:,-1]-time_beginning_2)<4e-3)[0][0]
    position_beginning_2 = np.rad2deg(estimated_shoulder_state[index_beginning_2, [0, 2]])

    time_beginning_3 = optimal_trajectory[12,-1]
    index_beginning_3 = np.where(abs(estimated_shoulder_state[:,-1]-time_beginning_3)<4e-3)[0][0]
    position_beginning_3 = np.rad2deg(estimated_shoulder_state[index_beginning_3, [0, 2]])

    position_end = np.rad2deg(estimated_shoulder_state[-1, [0, 2]])


    shortest_path_1 = np.vstack((np.linspace(position_beginning_1[0], position_beginning_2[0], 51), np.linspace(position_beginning_1[1], position_beginning_2[1], 51)))
    real_path_1 = np.rad2deg(np.vstack((estimated_shoulder_state[index_beginning_1:index_beginning_2, 0], estimated_shoulder_state[index_beginning_1:index_beginning_2, 2])))
    new_indices = np.linspace(0, real_path_1.shape[1]-1, num_samples).astype(int)
    real_path_1_sampled = real_path_1[:, new_indices]   # resample the real path to get only 51 points out of it
    index_shortest_path_1_on_strainmap = np.around(np.vstack(((shortest_path_1[0,:] - min_pe)/step, (shortest_path_1[1,:] - min_se)/step)), 0).astype(int)
    index_real_path_1_on_strainmap = np.around(np.vstack(((real_path_1_sampled[0,:] - min_pe)/step, (real_path_1_sampled[1,:] - min_se)/step)), 0).astype(int)

    strain_shortest_path_1 = np.transpose(strainmap[index_shortest_path_1_on_strainmap[0,:], index_shortest_path_1_on_strainmap[1,:]])
    strain_real_path_1 = np.transpose(strainmap[index_real_path_1_on_strainmap[0,:], index_real_path_1_on_strainmap[1,:]])

    shortest_path_2 = np.vstack((np.linspace(position_beginning_2[0], position_beginning_3[0], 51), np.linspace(position_beginning_2[1], position_beginning_3[1], 51)))
    real_path_2 = np.rad2deg(np.vstack((estimated_shoulder_state[index_beginning_2:index_beginning_3, 0], estimated_shoulder_state[index_beginning_2:index_beginning_3, 2])))
    new_indices = np.linspace(0, real_path_2.shape[1]-1, num_samples).astype(int)
    real_path_2_sampled = real_path_2[:, new_indices]   # resample the real path to get only 51 points out of it
    index_shortest_path_2_on_strainmap = np.around(np.vstack(((shortest_path_2[0,:] - min_pe)/step, (shortest_path_2[1,:] - min_se)/step)), 0).astype(int)
    index_real_path_2_on_strainmap = np.around(np.vstack(((real_path_2_sampled[0,:] - min_pe)/step, (real_path_2_sampled[1,:] - min_se)/step)), 0).astype(int)

    strain_shortest_path_2 = np.transpose(strainmap[index_shortest_path_2_on_strainmap[0,:], index_shortest_path_2_on_strainmap[1,:]])
    strain_real_path_2 = np.transpose(strainmap[index_real_path_2_on_strainmap[0,:], index_real_path_2_on_strainmap[1,:]])

    shortest_path_3 = np.vstack((np.linspace(position_beginning_3[0], position_end[0], 51), np.linspace(position_beginning_3[1], position_end[1], 51)))
    real_path_3 = np.rad2deg(np.vstack((estimated_shoulder_state[index_beginning_3:, 0], estimated_shoulder_state[index_beginning_3:, 2])))
    new_indices = np.linspace(0, real_path_3.shape[1]-1, num_samples).astype(int)
    real_path_3_sampled = real_path_3[:, new_indices]   # resample the real path to get only 51 points out of it
    index_shortest_path_3_on_strainmap = np.around(np.vstack(((shortest_path_3[0,:] - min_pe)/step, (shortest_path_3[1,:] - min_se)/step)), 0).astype(int)
    index_real_path_3_on_strainmap = np.around(np.vstack(((real_path_3_sampled[0,:] - min_pe)/step, (real_path_3_sampled[1,:] - min_se)/step)), 0).astype(int)

    strain_shortest_path_3 = np.transpose(strainmap[index_shortest_path_3_on_strainmap[0,:], index_shortest_path_3_on_strainmap[1,:]])
    strain_real_path_3 = np.transpose(strainmap[index_real_path_3_on_strainmap[0,:], index_real_path_3_on_strainmap[1,:]])

    strain_real_path = np.hstack((strain_real_path_1, strain_real_path_2, strain_real_path_3))
    strain_shortest_path = np.hstack((strain_shortest_path_1, strain_shortest_path_2, strain_shortest_path_3))

    # create the x axis with the correct timing
    x_axis = np.arange(len(strain_real_path))/len(strain_real_path)*(estimated_shoulder_state[-1, -1]-estimated_shoulder_state[index_beginning_1, -1])

    # strain along the optimal path (comprises also the "waiting phases" during which the optimization is performed)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x_axis, strain_real_path, label = 'Strain optimal path')
    ax.set_ylabel('Strain [%]')
    ax.set_xlabel('Time [s]')
    ax.legend()

    # strain along the shortest path, equivalent to traveling directly from start to goal every time (here no "waiting phase" is accounted for)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x_axis, strain_shortest_path, label = 'Strain shortest path')
    ax.set_ylabel('Strain [%]')
    ax.set_xlabel('Time [s]')   
    ax.legend()
    
    # visualize the human trajectory along the individual human DoF
    reference_se = np.hstack((optimal_trajectory[0, :-1], optimal_trajectory[6, :-1], optimal_trajectory[12, :-1]))
    time_est = np.linspace(estimated_shoulder_state[0,-1], estimated_shoulder_state[-1,-1], len(estimated_shoulder_state[:,-1]))
    time_ref = np.linspace(optimal_trajectory[0, -1], optimal_trajectory[12, -1], len(optimal_trajectory[0,:-1])*3)
    # plotting the reference is wrong since we do not account for the dead moments in which we wait for the new reference....

    fig = plt.figure()
    ax = fig.add_subplot(321)
    ax.plot(time_est, np.rad2deg(estimated_shoulder_state[:, 0]), label='estimated')
    ax.plot(time_ref, np.rad2deg(reference_se))
    ax.set_ylabel("PE [°]")
    ax.legend()
    ax = fig.add_subplot(323)
    ax.plot(estimated_shoulder_state[:,-1], np.rad2deg(estimated_shoulder_state[:, 2]))
    ax.set_ylabel("Estimated SE[°]")
    ax = fig.add_subplot(325)
    ax.plot(estimated_shoulder_state[:,-1],np.rad2deg(estimated_shoulder_state[:, 4]))
    ax.set_ylabel("Estimated AR [°]")
    ax.set_xlabel("Time [@ 10Hz]")
    ax = fig.add_subplot(322)
    ax.plot(estimated_shoulder_state[:,-1],np.rad2deg(estimated_shoulder_state[:, 1]))
    ax.set_ylabel("Estimated PE_dot [°/s]")
    ax = fig.add_subplot(324)
    ax.plot(estimated_shoulder_state[:,-1],np.rad2deg(estimated_shoulder_state[:, 3]))
    ax.set_ylabel("Estimated SE_dot [°/s]")
    ax = fig.add_subplot(326)
    ax.plot(estimated_shoulder_state[:,-1],np.rad2deg(estimated_shoulder_state[:, 5]))
    ax.set_ylabel("Estimated AR_dot [°/s]")
    ax.set_xlabel("Time [@ 10Hz]")
    ax.legend()
    fig.suptitle("EXP1: individual shoulder angles (with human)")


    # visualize 3D trajectory for the EE position
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyz_curr[:,0], xyz_curr[:,1], xyz_curr[:,2], label='Actual trajectory')
    ax.scatter(xyz_cmd[:,0], xyz_cmd[:,1], xyz_cmd[:,2], label='Reference trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title("EXP1: 3D EE trajectories (with human)")

    # visualize individual XYZ component of EE pose
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,0], label = 'x_curr', color='black', linestyle='dashed')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,0], label = 'x_cmd', color = 'black')
    ax.set_ylim(-0.7, -0.5)
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig.add_subplot(312)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,1], label = 'y_curr', color = 'black', linestyle='dashed')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,1], label = 'y_cmd', color = 'black')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig.add_subplot(313)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,2], label = 'z_curr', color = 'black', linestyle='dashed')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,2], label = 'z_cmd', color = 'black')
    if z_uncompensated is not None:
        ax.plot(z_uncompensated[:,-1], z_uncompensated[:,0], label = 'z_des', color='cornflowerblue')
    ax.set_ylabel('[m]')
    ax.set_xlabel('time [s]')
    ax.legend()
    fig.suptitle("EXP5: EE cartesian position")

    # visualize orientation mismatch between commanded and executed motion

    # orientation_mismatch = np.zeros((angvec_cmd.shape[0],1))
    # for index in range(angvec_cmd.shape[0]):
    #     orientation_mismatch[index] = np.arccos(np.dot(angvec_cmd[index, 1:4], angvec_curr[index, 1:4])/(np.linalg.norm(angvec_cmd[index, 1:4])*np.linalg.norm(angvec_curr[index, 1:4])))
    # Calculate the absolute differences between each timestamp in angvec_cmd and angvec_curr
    time_cmd = angvec_cmd[:,-1]
    time_curr = angvec_curr[:,-1]
    abs_diff = np.abs(time_cmd[:, None] - time_curr)
    # Find the indices of the minimum absolute differences for each timestamp in angvec_cmd
    nearest_indices = np.argmin(abs_diff, axis=1)

    # Select the rows from angvec_curr using the nearest indices
    angvec_curr_selected = angvec_curr[nearest_indices]
    
    orientation_mismatch = np.rad2deg(np.arccos(np.sum(angvec_cmd[:, 1:4] * angvec_curr_selected[:, 1:4], axis=1) /
                                  (np.linalg.norm(angvec_cmd[:, 1:4], axis=1) * np.linalg.norm(angvec_curr_selected[:, 1:4], axis=1))))

    angle_mismatch = angvec_cmd[:, 0] - angvec_curr_selected[:, 0]
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(angvec_cmd[:, -1], orientation_mismatch, label='axis mismatch')
    ax.set_ylabel('[deg]')
    ax.legend()
    ax = fig.add_subplot(212)
    ax.plot(angvec_curr_selected[:, -1], angle_mismatch, label='angle mismatch')
    ax.set_ylabel('[deg]')
    ax.set_xlabel('time [s]')
    ax.legend()
    fig.suptitle("EXP5: EE orientation")

    # retrieve the accelerations that the arm will undergo following optimal trajectory
    time_diff = 0.1
    optimal_trajectory_dot = np.gradient(optimal_trajectory[:,:-1], time_diff, axis=1)
    optimal_pe_ddot = optimal_trajectory_dot[1,:]
    optimal_se_ddot = optimal_trajectory_dot[3,:]
    for index_optim in range(1, num_optim):
        optimal_pe_ddot = np.hstack((optimal_pe_ddot, optimal_trajectory_dot[6*index_optim+1,:]))
        optimal_se_ddot = np.hstack((optimal_se_ddot, optimal_trajectory_dot[6*index_optim+3,:]))

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(np.rad2deg(optimal_pe_ddot))
    ax.set_ylabel('[deg/(s^2)]')
    ax.set_title('PE acceleration')
    ax = fig.add_subplot(212)
    ax.plot(np.rad2deg(optimal_se_ddot))
    ax.set_ylabel('[deg/(s^2)]')
    ax.set_title('SE acceleration')

    # visualize magnitude of interaction force
    if interaction_force_magnitude is not None:
        fs = 1/(interaction_force_magnitude[1,-1] - interaction_force_magnitude[0, -1])
        cutoff = 20
        order = 2
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        interaction_force_magnitude_filt = filtfilt(b, a, interaction_force_magnitude[:,0])

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(interaction_force_magnitude[:,-1], interaction_force_magnitude[:,0])
        ax.plot(interaction_force_magnitude[:,-1], interaction_force_magnitude_filt)
        ax.set_ylabel('[N]')
        ax.set_title('Interaction force')

    data = {}
    data['optimal_trajectory'] = optimal_trajectory
    data['optimal_trajectory_dot'] = optimal_trajectory_dot
    data['optimal_controls'] = None

    file_path = os.path.join(path_to_bag, 'data.pkl')
    with open(file_path, "wb") as file:
        # Use pickle to dump the velocity data to the file
        pickle.dump(data, file)

    plt.show(block=True)

if __name__ == '__main__':
    main()