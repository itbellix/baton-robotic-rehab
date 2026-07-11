"""
Script to analyze the data collected in a rosbag during the real robot experiments and varying activation.
This script produces the results that I aggregated in Fig. 10 of the paper (left side).
"""

import os
import pickle
from spatialmath import SO3
import numpy as np
import rosbag
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

num_params = 6
PLOT_ESTIMATED_FULL_STATE = False
PLOT_EE_TRAJECTORY = False
PLOT_MUSCLE_ACTIVATION = True
PLOT_INTERACTION_FORCE = False
PLOT_TRAJECTORY_ON_STRAINMAP = True
PLOT_2D_TRAJECTORY = True

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

def colored_line(ax, x, y, c, cmap, norm, linestyle='solid', linewidth=2.5, label=None):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c[:-1])
    lc.set_linewidth(linewidth)
    lc.set_linestyle(linestyle)

    ax.add_collection(lc)

    # Optional legend entry
    if label is not None:
        ax.plot([], [], linestyle=linestyle,
                color=cmap(norm(np.max(c))),
                label=label)

    return lc

def process_trial(subject, trial, path_to_bag, file_strainmaps, time_0_percent, time_final_percent):
    # select the appropriate bag file depending on the subject and trial considered
    if subject == 1:
        bag_file_name = f'exp11_stijn_{trial}.bag'

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
        interaction_torque_x = None
        for _, msg, time_msg in bag.read_messages(topics=['/compensated_wrench']):
            time_curr = time_msg.to_time()
            if interaction_torque_x is None:
                interaction_torque_x = np.hstack((msg.data[3], time_curr))
            else:
                interaction_torque_x = np.vstack((interaction_torque_x, np.hstack((msg.data[3], time_curr))))

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
    init_time = xyz_curr[int(xyz_curr.shape[0]/100*time_0_percent), -1]         # identify initial timestep
    end_time = xyz_curr[int(xyz_curr.shape[0]/100*time_final_percent), -1] - init_time # identify final timestep

    estimated_shoulder_state[:,-1] = estimated_shoulder_state[:,-1] - init_time   # center time values starting at initial time
    xyz_curr[:,-1] = xyz_curr[:,-1] - init_time
    xyz_cmd[:,-1] = xyz_cmd[:,-1] - init_time
    if z_uncompensated is not None:
        z_uncompensated[:,-1] = z_uncompensated[:,-1] - init_time
    angvec_curr[:,-1] = angvec_curr[:,-1] - init_time
    angvec_cmd[:,-1] = angvec_cmd[:,-1] - init_time
    optimal_trajectory[:,:,-1] = optimal_trajectory[:,:,-1] - init_time
    if interaction_torque_x is not None:
        interaction_torque_x[:,-1] = interaction_torque_x[:,-1] - init_time
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

    if muscle_activation_max is not None:
        muscle_activation_max = muscle_activation_max[(muscle_activation_max[:,-1]>0) & (muscle_activation_max[:,-1]<end_time)]
        muscle_activation_selected = muscle_activation_selected[(muscle_activation_selected[:,-1]>0) & (muscle_activation_selected[:,-1]<end_time)]

    if interaction_torque_x is not None:
        interaction_torque_x = interaction_torque_x[(interaction_torque_x[:,-1]>0) & (interaction_torque_x[:,-1]<end_time)]

    filtered_trajectories = []                      # also filter the optimal trajectories
    for i in range(optimal_trajectory.shape[0]):
        slice_i = optimal_trajectory[i, :, :]
        filtered_slice = slice_i[(slice_i[:, -1] > 0) & (slice_i[:, -1] < end_time)]
        filtered_trajectories.append(filtered_slice)

    optimal_trajectory = np.stack(filtered_trajectories, axis=0)        # stack everything into a 3D array again

    strainmap = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[100, 4], 0)
    max_strain = 1.9    # play with this number a bit to get best visualizations

    # we want to extract the time axes from both the shoulder trajectory and the muscle activation estimates
    t_shoulder = estimated_shoulder_state[:, -1]
    t_muscle   = muscle_activation_selected[:, -1] if muscle_activation_selected is not None else None
    # share time normalization
    t_min = min(t_shoulder.min(), t_muscle.min() if t_muscle is not None else t_shoulder.min())
    t_max = max(t_shoulder.max(), t_muscle.max() if t_muscle is not None else t_shoulder.max())
    norm = Normalize(vmin=t_min, vmax=t_max)

    # selecting white -> dark blue colormap ---
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "darkblue"])
    
    if PLOT_TRAJECTORY_ON_STRAINMAP:
        # visualize 2D trajectory on strainmap
        start = np.array([60, 60])
        goal_1 = np.array([100, 90])
        fig = plt.figure()
        ax = fig.add_subplot()
        cb = ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=strainmap.min(), vmax=max_strain)
        ax.plot(np.rad2deg(estimated_shoulder_state[:, 0]), np.rad2deg(estimated_shoulder_state[:, 2]))
        ax.scatter(start[0], start[1], label = 'goal', c = 'green', edgecolors='black')
        ax.scatter(goal_1[0], goal_1[1], label = 'start', c='red', edgecolors='black')
        ax.set_ylim(50, 110)
        ax.set_xlim(45, 100)
        ax.set_xlabel("Plane of Elevation [°]")
        ax.set_ylabel("Shoulder Elevation [°]")
        ax.legend()
        ax.set_title(f"EXP11 (trial {trial}): Trajectory on strainmap (0 activation)")
        fig.colorbar(cb)

    if PLOT_ESTIMATED_FULL_STATE:
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
        fig.suptitle(f"EXP11 (trial {trial}): individual shoulder angles (with human)")

    if PLOT_EE_TRAJECTORY:
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
        fig.suptitle(f"EXP11 (trial {trial}): EE cartesian position")

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
        fig.suptitle(f"EXP11 (trial {trial}): EE orientation")

    if PLOT_INTERACTION_FORCE:
        # visualize magnitude of interaction force
        if interaction_torque_x is not None:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(interaction_torque_x[:,-1], interaction_torque_x[:,0])
            ax.set_ylabel('[N]')
            ax.set_title(f'Interaction torque (x) - trial {trial}')

    if PLOT_MUSCLE_ACTIVATION:
        if muscle_activation_selected is not None:
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.plot(muscle_activation_selected[:,-1], muscle_activation_selected[:,0], label='ISI')
            ax.legend()
            ax.set_ylabel('activation')
            ax.set_title(f'Muscle activation - trial {trial}')

            # print to screen the average maximum muscle activation
            print('Average maximum activation: ', np.mean(muscle_activation_max[:, 0]))

    if PLOT_2D_TRAJECTORY:
        if muscle_activation_selected is not None:
            # now we want to plot plane of elevation and shoulder elevation over time, colored by the muscle activation of the selected muscle (ISI or SSPA)
            norm = Normalize(vmin=muscle_activation_selected[:, 0].min(),vmax=muscle_activation_selected[:, 0].max())
            red_cmap = LinearSegmentedColormap.from_list("dark_to_bright_red",["#4d0000", "#ff0000"])   # dark red → bright red
            with_cmap = 0
            fig, ax = plt.subplots(figsize=(16, 4))

            # X/Y trajectory (adapt if needed)
            time = estimated_shoulder_state[:, -1]
            pe = np.rad2deg(estimated_shoulder_state[:, 0])
            se = np.rad2deg(estimated_shoulder_state[:, 2])

            if with_cmap:
                # --- Solid line (first trajectory) ---
                lc1 = colored_line(
                    ax, time, pe, muscle_activation_selected[:, 0],
                    cmap=red_cmap,
                    norm=norm,
                    linestyle='solid',
                    label='PE'
                )

                # --- Dashed line (second trajectory) ---
                lc2 = colored_line(
                    ax, time, se, muscle_activation_selected[:, 0],
                    cmap=red_cmap,
                    norm=norm,
                    linestyle='dashed',
                    label='SE'
                )

                # Custom legend
                legend_items = [
                    Line2D([0], [0], color="#ff0000", lw=2.5, linestyle='solid', label='PE'),
                    Line2D([0], [0], color="#ff0000", lw=2.5, linestyle='dashed', label='SE')
                ]

            else:
                ax.plot(time, pe, label='PE', color='black', linestyle='solid')
                ax.plot(time, se, label='SE', color='black', linestyle='dashed')
                legend_items = [
                    Line2D([0], [0], color='black', lw=2.5, linestyle='solid', label='PE'),
                    Line2D([0], [0], color='black', lw=2.5, linestyle='dashed', label='SE')
                ]

            # Markers (optional)
            ax.scatter(0, start[0], label='goal_PE', c='green', edgecolors='black', zorder=5)
            ax.scatter(0, start[1], label='goal_SE', c='green', edgecolors='black', zorder=5)
            ax.scatter(0, goal_1[0], label='start', c='red', edgecolors='black', zorder=5)
            ax.scatter(0, goal_1[1], label='start', c='red', edgecolors='black', zorder=5)

            ax.set_xlabel("time [s]")
            ax.set_ylabel("DoF [°]")

            # Activation colorbar
            if with_cmap:
                sm = plt.cm.ScalarMappable(cmap=red_cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label="Muscle activation")

            # Custom legend
            ax.legend(handles=legend_items)
            ax.set_title(f'DoFs - trial {trial}')


def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_11')

    # define initial and final times in percentage of the overall duration
    time_0_percent = 0
    time_final_percent = 99

    # load the strainmap dictionary used in the experiment
    # file_strainmaps = '/home/itbellix/Desktop/GitHub/PTbot_official/Personal_Results/Strains/Passive/AllMuscles/params_strainmaps_num_Gauss_3/params_strainmaps_num_Gauss_3.pkl' 
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models', 'Strain Maps', 'Active', 'differentiable_strainmaps_ISI.pkl')

    # select the subject and the trial(s) to analyze
    subject = 1 # available 1
    trials = [1, 2, 3, 5, 6]   # available 1, 2, 3, 4, 5, 6 (subject 1)
    # good appear to be 1, 2 (some parts), 3 (only first trajectory), 5, 6

    # allow a single trial (e.g. trials = 1) as well as a list of trials
    if np.isscalar(trials):
        trials = [trials]

    # produce all the plots for each trial in sequence
    for trial in trials:
        print(f"\n===== Processing subject {subject}, trial {trial} =====")
        process_trial(subject, trial, path_to_bag, file_strainmaps, time_0_percent, time_final_percent)

    # show all the produced figures at once
    plt.show(block=True)


if __name__ == '__main__':
    main()