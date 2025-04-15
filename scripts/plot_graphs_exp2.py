"""
Script to analyze the data collected in a rosbag during the real robot experiments and varying activation.
This script produces the results that I aggregated in Fig. 9 of the paper.
"""

import os
import pickle
from spatialmath import SO3
import numpy as np
import rosbag
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

num_params = 6
def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    '''
    Function used for the 2D interpolation
    '''
    return amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2)+(y-y0)**2/(2*sigma_y**2)))+offset

def generate_approximated_strainmap(file_strainmaps, ar_value, act_value = 0):
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

    act_min = 0.0
    act_step = 0.005

    step_viz = 0.05
    step_from_model = 4

    # create the grids that will be used for interpolation (and visualization)
    pe_datapoints = np.array(np.arange(min_pe, max_pe, step_viz))
    se_datapoints = np.array(np.arange(min_se, max_se, step_viz))
    ar_datapoints = np.array(np.arange(min_ar, max_ar, step_from_model))

    X,Y = np.meshgrid(pe_datapoints, se_datapoints, indexing='ij')
    X_norm = X/max_pe
    Y_norm = Y/max_se
    data = np.vstack((X.ravel(), Y.ravel()))
    data_norm = np.vstack((X_norm.ravel(), Y_norm.ravel()))

    index_strainmap_current = np.argmin(np.abs(ar_datapoints - np.rad2deg(ar_value)))   # index of the strainmap
    act_index = int(np.floor((act_value - act_min)/act_step))

    # check if we have data for varying activation or not
    if isinstance(strainmaps_dict['all_params_gaussians'][0], np.ndarray):
        # fixed activation
        params_gaussians = strainmaps_dict['all_params_gaussians'][index_strainmap_current]
        num_gaussians = strainmaps_dict['num_gaussians'][index_strainmap_current]
    if isinstance(strainmaps_dict['all_params_gaussians'][0], list):
        # varying activation
        print("remember to check the strainmap that you want to visualize!")
        params_gaussians = strainmaps_dict['all_params_gaussians'][index_strainmap_current][act_index]
        num_gaussians = strainmaps_dict['num_gaussians'][index_strainmap_current][act_index]

    fit = np.zeros((pe_datapoints.shape[0], se_datapoints.shape[0]))
    for i in range(len(params_gaussians)//num_params):
        fit += gaussian_2d(X_norm, Y_norm, *params_gaussians[i*num_params:i*num_params+num_params])

    return fit



def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_2')
    # bag_file_name = '2024-05-07-19-05-30_exp2.bag'      # for experiment without activation
    bag_file_name = 'exp2_passive_ISI.bag'    # for experiment with activation

    # load the strainmap dictionary used in the experiment
    # file_strainmaps = '/home/itbellix/Desktop/GitHub/PTbot_official/Personal_Results/Strains/Passive/AllMuscles/params_strainmaps_num_Gauss_3/params_strainmaps_num_Gauss_3.pkl' 
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models', 'Strain Maps', 'Active', 'differentiable_strainmaps_ISI.pkl')

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
        optimal_trajectory = None
        optimal_strain = None
        for _, msg, time_msg in bag.read_messages(topics=['/optimization_output']):
            data = np.reshape(msg.data[0:-1], (9, 11))
            time_curr = time_msg.to_time()
            if optimal_strain is None:
                optimal_strain = np.hstack((data[8,:], time_curr))
                optimal_trajectory = np.hstack((data[0:6,:], time_curr * np.ones((6,1))))
            else:
                optimal_strain = np.vstack((optimal_strain, np.hstack((data[8,:], time_curr))))
                optimal_trajectory = np.vstack((optimal_trajectory, np.hstack((data[0:6,:], time_curr * np.ones((6,1))))))

        print('Extracting forces')
        interaction_force_magnitude = None
        for _, msg, time_msg in bag.read_messages(topics=['/ft_sensor_data']):
            time_curr = time_msg.to_time()
            if interaction_force_magnitude is None:
                interaction_force_magnitude = np.hstack((np.linalg.norm(msg.data[0:3]), time_curr))
            else:
                interaction_force_magnitude = np.vstack((interaction_force_magnitude, np.hstack((np.linalg.norm(msg.data[0:3]), time_curr))))

        print('Extracting muscle activation')
        num_muscles = 22
        index_muscle = 11       # 11 for infraspinatus inferior (ISI), 18 for supraspinatus anterior (SSPA)
        muscle_activation_max = None
        muscle_activation_selected = None
        for _, msg, time_msg in bag.read_messages(topics=['/estimated_muscle_activation']):
            time_curr = time_msg.to_time()
            if muscle_activation_max is None:
                muscle_activation_max = np.hstack((np.max(msg.data[0:num_muscles]), time_curr))
                muscle_activation_selected = np.hstack((msg.data[index_muscle], time_curr))
            else:
                muscle_activation_max = np.vstack((muscle_activation_max, np.hstack((np.max(msg.data[0:num_muscles]), time_curr))))
                muscle_activation_selected = np.vstack((muscle_activation_selected, np.hstack((msg.data[index_muscle], time_curr))))


    # reshaping the optimal trajectories into 3D array
    optimal_trajectory = np.reshape(optimal_trajectory, (6, -1, optimal_trajectory.shape[1]), order='F')

    # Now, let's filter the data to retain only the interesting part of the experiment
    # (i.e., when the subject is wearing the brace properly and the robot is moving)
    init_time = xyz_curr[int(xyz_curr.shape[0]/100*1), -1]         # identify initial timestep
    end_time = xyz_curr[int(xyz_curr.shape[0]/100*99), -1] - init_time # identify final timestep

    estimated_shoulder_state[:,-1] = estimated_shoulder_state[:,-1] - init_time   # center time values starting at initial time
    xyz_curr[:,-1] = xyz_curr[:,-1] - init_time
    xyz_cmd[:,-1] = xyz_cmd[:,-1] - init_time
    if z_uncompensated is not None:
        z_uncompensated[:,-1] = z_uncompensated[:,-1] - init_time
    angvec_curr[:,-1] = angvec_curr[:,-1] - init_time
    angvec_cmd[:,-1] = angvec_cmd[:,-1] - init_time
    optimal_trajectory[:,:,-1] = optimal_trajectory[:,:,-1] - init_time
    if interaction_force_magnitude is not None:
        interaction_force_magnitude[:,-1] = interaction_force_magnitude[:,-1] - init_time
    if muscle_activation_max is not None:
        muscle_activation_max[:,-1] = muscle_activation_max[:,-1] - init_time
        muscle_activation_selected[:,-1] = muscle_activation_selected[:,-1] - init_time

    estimated_shoulder_state = estimated_shoulder_state[(estimated_shoulder_state[:,-1]>0) & (estimated_shoulder_state[:,-1]<end_time)]
    xyz_curr = xyz_curr[(xyz_curr[:,-1]>0) & (xyz_curr[:,-1]<end_time)]    # retain data after initial time
    xyz_cmd = xyz_cmd[(xyz_cmd[:,-1]>0) & (xyz_cmd[:,-1]<end_time)]
    if z_uncompensated is not None:
        z_uncompensated = z_uncompensated[(z_uncompensated[:,-1]>0) & (z_uncompensated[:,-1]<end_time)]
    angvec_curr = angvec_curr[(angvec_curr[:,-1]>0) & (angvec_curr[:,-1]<end_time)]
    angvec_cmd = angvec_cmd[(angvec_cmd[:,-1]>0) & (angvec_cmd[:,-1]<end_time)]

    filtered_trajectories = []                      # also filter the optimal trajectories
    for i in range(optimal_trajectory.shape[0]):
        slice_i = optimal_trajectory[i, :, :]
        filtered_slice = slice_i[(slice_i[:, -1] > 0) & (slice_i[:, -1] < end_time)]
        filtered_trajectories.append(filtered_slice)

    optimal_trajectory = np.stack(filtered_trajectories, axis=0)        # stack everything into a 3D array again

    strainmap_act0_0 = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[100, 4])
    

    # visualize 2D trajectory on strainmap (zero activation)    USING bag_file_name = '2024-05-07-19-05-30_exp2.bag'
    len_data = len(estimated_shoulder_state)
    begin_d = int(len_data/100*0.01)
    end_d = int(len_data/100*29.0)

    fig = plt.figure()
    ax = fig.add_subplot()
    cb = ax.imshow(strainmap_act0_0.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=strainmap_act0_0.min(), vmax=0.5)
    ax.plot(np.rad2deg(estimated_shoulder_state[begin_d:end_d, 0]), np.rad2deg(estimated_shoulder_state[begin_d:end_d, 2]))
    ax.scatter(np.array([45]), np.array([95]), label = 'goal', c = 'green', edgecolors='black')
    ax.scatter(np.array([60]), np.array([60]), label = 'start', c='red', edgecolors='black')
    ax.set_ylim(45, 110)
    ax.set_xlim(30, 80)
    ax.set_xlabel("Plane of Elevation [°]")
    ax.set_ylabel("Shoulder Elevation [°]")
    ax.legend()
    ax.set_title("EXP2: Trajectory on strainmap (0 activation)")
    fig.colorbar(cb)

    # visualize 2D trajectory on strainmap
    strainmap = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[100, 4])

    # visualize 2D trajectory on strainmap
    fig = plt.figure()
    ax = fig.add_subplot()
    # cb =ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=0, vmax=strainmap.max())
    # fig.colorbar(cb, ax=ax, label = 'Strain [%]')
    ax.plot(np.rad2deg(estimated_shoulder_state[:-1, 0]), np.rad2deg(estimated_shoulder_state[:-1, 2]))
    ax.scatter(np.array([50]), np.array([100]), label = 'start')
    ax.scatter(np.array([100]), np.array([100]), label = 'goal1')
    ax.scatter(np.array([50]), np.array([70]), label = 'goal2')
    ax.scatter(np.array([60]), np.array([110]), label = 'goal3', c='black')

    # plot also the optimal trajectories
    np.rad2deg(estimated_shoulder_state[begin_d:end_d, 0]), np.rad2deg(estimated_shoulder_state[begin_d:end_d, 2])
    ax.set_xlabel("Plane of elevation [deg]")
    ax.set_ylabel("Shoulder elevation [deg]")
    ax.legend()
    ax.set_title("EXP1: Trajectory on strainmap (with human)")

    # visualize individual human DoF
    fig = plt.figure()
    ax = fig.add_subplot(321)
    ax.plot(np.rad2deg(estimated_shoulder_state[:, 0]))
    ax.set_ylabel("Estimated PE [°]")
    ax = fig.add_subplot(323)
    ax.plot(np.rad2deg(estimated_shoulder_state[:, 2]))
    ax.set_ylabel("Estimated SE[°]")
    ax = fig.add_subplot(325)
    ax.plot(np.rad2deg(estimated_shoulder_state[:, 4]))
    ax.set_ylabel("Estimated AR [°]")
    ax.set_xlabel("Time [@ 10Hz]")
    ax = fig.add_subplot(322)
    ax.plot(np.rad2deg(estimated_shoulder_state[:, 1]))
    ax.set_ylabel("Estimated PE_dot [°/s]")
    ax = fig.add_subplot(324)
    ax.plot(np.rad2deg(estimated_shoulder_state[:, 3]))
    ax.set_ylabel("Estimated SE_dot [°/s]")
    ax = fig.add_subplot(326)
    ax.plot(np.rad2deg(estimated_shoulder_state[:, 5]))
    ax.set_ylabel("Estimated AR_dot [°/s]")
    ax.set_xlabel("Time [@ 10Hz]")
    ax.legend()
    fig.suptitle("EXP2: individual shoulder angles (with human)")

    # visualize 3D trajectory for the EE position
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xyz_curr[:,0], xyz_curr[:,1], xyz_curr[:,2], label='Actual trajectory')
    ax.scatter(xyz_cmd[:,0], xyz_cmd[:,1], xyz_cmd[:,2], label='Reference trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    # visualize individual XYZ component of EE pose
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,0], label = 'x_curr', color='black')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,0], label = 'x_cmd', color='black', linestyle='dashed')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig.add_subplot(312)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,1], label = 'y_curr', color='black')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,1], label = 'y_cmd', color='black', linestyle='dashed')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig.add_subplot(313)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,2], label = 'z_curr', color='black')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,2], label = 'z_cmd', color='black', linestyle='dashed')
    if z_uncompensated is not None:
        ax.plot(z_uncompensated[:,-1], z_uncompensated[:,0], label = 'z_des', color='cornflowerblue', linestyle='dashed')
    ax.set_ylabel('[m]')
    ax.legend()
    fig.suptitle("EXP2: EE cartesian position")

    # visualize orientation mismatch between commanded and executed motion
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
    fig.suptitle("EXP2: EE orientation")

    # visualize magnitude of interaction force
    if interaction_force_magnitude is not None:
        fs = 1/(interaction_force_magnitude[1,-1] - interaction_force_magnitude[0, -1])
        cutoff = 30
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

    if muscle_activation_selected is not None:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(muscle_activation_selected[:,-1], muscle_activation_selected[:,0])
        ax.set_ylabel('activation')
        ax.set_title('Supraspinatus Anterior')

        # print to screen the average maximum muscle activation
        print('Average maximum activation: ', np.mean(muscle_activation_max[:, 0]))
    
    plt.show(block=True)


if __name__ == '__main__':
    main()