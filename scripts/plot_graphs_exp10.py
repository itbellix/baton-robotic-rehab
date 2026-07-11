"""
Script to analyze the data collected in a rosbag during the real robot experiments and varying activation.
Strain map for SSPA muscle navigated by ISI activation. The results produced here are aggregated in Fig. 10 of the paper (right side)
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
MOVEMENT_SAMPLES = 1000     # samples used to resample each trial over the 0-100% movement phase

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


def resample_to_grid(x, y, grid):
    """Interpolate y(x) onto the given grid. x is sorted first to guarantee monotonicity."""
    order = np.argsort(x, kind='stable')
    return np.interp(grid, x[order], y[order])


def process_trial(subject, trial, path_to_bag, file_strainmaps, intervals):
    # select the appropriate bag file depending on the subject and trial considered
    if subject == 1:
        bag_file_name = f'exp10_stijn_{trial}.bag'

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
        muscle_activation_selected_2 = None
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

    # Now, let's retain only the two movement intervals defined for this trial.
    # Each interval is a (start%, end%) window of the FULL experiment duration. The data of
    # the first interval is remapped to 0-50% of the displayed movement, the data of the
    # second interval to 50-100%. The two segments are then concatenated: since we are only
    # interested in time-independent quantities, the segments having different timestamps
    # (and possibly different durations) is not a problem.
    (first_lo, first_hi), (second_lo, second_hi) = intervals

    # absolute timestamp (taken from xyz_curr) corresponding to a percentage of the recording
    def abs_time_at(pct):
        idx = int(np.clip(round(xyz_curr.shape[0] / 100 * pct), 0, xyz_curr.shape[0] - 1))
        return xyz_curr[idx, -1]

    t1_start, t1_end = abs_time_at(first_lo), abs_time_at(first_hi)
    t2_start, t2_end = abs_time_at(second_lo), abs_time_at(second_hi)

    # keep the rows (last column = timestamp) falling within the two intervals, remap their
    # timestamps to 0-50% (first interval) and 50-100% (second interval), then concatenate
    def segment_and_remap(arr):
        if arr is None:
            return None
        t = arr[:, -1]
        seg1 = arr[(t >= t1_start) & (t <= t1_end)].copy()
        seg2 = arr[(t >= t2_start) & (t <= t2_end)].copy()
        seg1[:, -1] = (seg1[:, -1] - t1_start) / (t1_end - t1_start) * 50
        seg2[:, -1] = 50 + (seg2[:, -1] - t2_start) / (t2_end - t2_start) * 50
        return np.vstack((seg1, seg2))

    estimated_shoulder_state = segment_and_remap(estimated_shoulder_state)
    xyz_curr = segment_and_remap(xyz_curr)
    xyz_cmd = segment_and_remap(xyz_cmd)
    z_uncompensated = segment_and_remap(z_uncompensated)
    angvec_curr = segment_and_remap(angvec_curr)
    angvec_cmd = segment_and_remap(angvec_cmd)
    interaction_torque_x = segment_and_remap(interaction_torque_x)
    if muscle_activation_max is not None:
        muscle_activation_max = segment_and_remap(muscle_activation_max)
        muscle_activation_selected = segment_and_remap(muscle_activation_selected)

    # the optimal trajectories live in a 3D array (n_dof, n_samples, n_cols): remap each
    # slice independently and stack them back together
    optimal_trajectory = np.stack(
        [segment_and_remap(optimal_trajectory[i]) for i in range(optimal_trajectory.shape[0])],
        axis=0
    )

    # pick a representative axial-rotation value to slice the strainmap (index clamped in case
    # the concatenated segments contain fewer than 100 samples)
    ar_index = min(100, estimated_shoulder_state.shape[0] - 1)
    strainmap = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[ar_index, 4], 0)
    max_strain = 1.9    # play with this number a bit to get best visualizations
    

    if PLOT_TRAJECTORY_ON_STRAINMAP:
        # visualize 2D trajectory on strainmap
        start = np.array([60, 60])
        goal_1 = np.array([45, 95])
        fig = plt.figure()
        ax = fig.add_subplot()
        cb = ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=strainmap.min(), vmax=max_strain)
        ax.plot(np.rad2deg(estimated_shoulder_state[:, 0]), np.rad2deg(estimated_shoulder_state[:, 2]))
        ax.scatter(start[0], start[1], label = 'goal', c = 'green', edgecolors='black')
        ax.scatter(goal_1[0], goal_1[1], label = 'start', c='red', edgecolors='black')
        ax.set_ylim(45, 105)
        ax.set_xlim(35, 80)
        ax.set_xlabel("Plane of Elevation [°]")
        ax.set_ylabel("Shoulder Elevation [°]")
        ax.legend()
        ax.set_title(f"EXP10 (trial {trial}): Trajectory on strainmap (0 activation)")
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
        fig.suptitle(f"EXP10 (trial {trial}): individual shoulder angles (with human)")

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
        ax.set_xlabel('Movement [%]')
        ax.legend()
        fig.suptitle(f"EXP10 (trial {trial}): EE cartesian position")

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
        ax.set_xlabel('Movement [%]')
        ax.legend()
        fig.suptitle(f"EXP10 (trial {trial}): EE orientation")

    if PLOT_INTERACTION_FORCE:
        # visualize magnitude of interaction force
        if interaction_torque_x is not None:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(interaction_torque_x[:,-1], interaction_torque_x[:,0])
            ax.set_ylabel('[N]')
            ax.set_xlabel('Movement [%]')
            ax.set_title(f'Interaction torque (x) - trial {trial}')

    # Resample the signals of interest onto a common movement-percentage grid (0-100%), so
    # that every trial has the same number of samples and can be overlaid on a single figure.
    # The overlaid muscle-activation and 2D-DoF figures are produced once, across all trials,
    # in main().
    movement_grid = np.linspace(0, 100, MOVEMENT_SAMPLES)
    pe_grid = resample_to_grid(estimated_shoulder_state[:, -1], np.rad2deg(estimated_shoulder_state[:, 0]), movement_grid)
    se_grid = resample_to_grid(estimated_shoulder_state[:, -1], np.rad2deg(estimated_shoulder_state[:, 2]), movement_grid)

    activation_isi = None
    activation_sspa = None
    if muscle_activation_selected is not None:
        activation_isi = resample_to_grid(muscle_activation_selected[:, -1], muscle_activation_selected[:, 0], movement_grid)
        if muscle_activation_selected_2 is not None:
            activation_sspa = resample_to_grid(muscle_activation_selected_2[:, -1], muscle_activation_selected_2[:, 0], movement_grid)
        # print to screen the average maximum muscle activation
        print('Average maximum activation: ', np.mean(muscle_activation_max[:, 0]))

    return {
        'trial': trial,
        'movement_pct': movement_grid,
        'pe': pe_grid,
        'se': se_grid,
        'activation_isi': activation_isi,
        'activation_sspa': activation_sspa,
    }


def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_10')

    # load the strainmap dictionary used in the experiment
    # file_strainmaps = '/home/itbellix/Desktop/GitHub/PTbot_official/Personal_Results/Strains/Passive/AllMuscles/params_strainmaps_num_Gauss_3/params_strainmaps_num_Gauss_3.pkl' 
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models', 'Strain Maps', 'Active', 'differentiable_strainmaps_SSPA.pkl')

    # select the subject and the trial(s) to analyze
    subject = 1 # available 1
    trials = [1, 3, 4, 5, 5, 6, 6, 7, 7]   # available 1, 2, 3, 4, 5, 6, 7 (subject 1)
    # good are trial 1, 3 (only first trajectory), 4?, 5, 6, 7

    # For each trial (positionally: entry i refers to trials[i]) define two intervals, given
    # as percentages of the FULL experiment duration. The data in the first interval is
    # remapped to 0-50% of the displayed movement, the data in the second interval to
    # 50-100%; the two segments are then concatenated for visualization.
    trial_times = [
        [(13, 24), (28, 45)],   # trial 1
        [(14, 26), (29.5, 45)], # trial 3
        [(13, 21), (26, 40)],   # trial 4
        [(16, 28), (33, 50)],   # trial 5 (first trajectory)
        [(53, 64), (70, 84)],   # trial 5 (second trajectory)
        [(15, 27), (31, 46)],   # trial 6 (first trajectory)
        [(51, 68), (72, 88)],   # trial 6 (second trajectory)
        [(14, 28), (30, 45)],   # trial 7 (first trajectory)
        [(61, 73), (77, 90)],   # trial 7 (second trajectory)
    ]

    # allow a single trial (e.g. trials = 1) as well as a list of trials
    if np.isscalar(trials):
        trials = [trials]

    assert len(trial_times) == len(trials), \
        "trial_times must have exactly one entry per trial in 'trials'"

    # Optionally restrict only the PLOTS to a subset of the (trial, intervals) pairs.
    # The SD analysis below always uses ALL pairs; this selection affects the overlaid
    # figures only. Selection is positional (0-based), so a trial that appears more than
    # once keeps the trial_times entry it is paired with. Set to None to plot all pairs.
    # e.g. selected = [0, 2, 8] plots the 1st, 3rd and 9th pairs (right part of figure 10).
    selected = [0, 2, 8]

    # process every trial, collecting the resampled curves for the overlaid figures
    results = []
    for trial, intervals in zip(trials, trial_times):
        (first_lo, first_hi), (second_lo, second_hi) = intervals
        print(f"\n===== Processing subject {subject}, trial {trial} "
              f"(intervals {first_lo}-{first_hi}% and {second_lo}-{second_hi}% of the recording) =====")
        results.append(process_trial(subject, trial, path_to_bag, file_strainmaps, intervals))

    # ---- aggregate variability metrics across trials (all resampled to the common grid) ----
    # muscle activation: SD across trials at each movement sample, then averaged
    activations = [res['activation_isi'] for res in results if res['activation_isi'] is not None]
    if len(activations) > 1:
        activations = np.vstack(activations)                    # (n_trials, MOVEMENT_SAMPLES)
        std_activation = np.std(activations, axis=0, ddof=1)    # (MOVEMENT_SAMPLES,)
        print(f"Mean SD of muscle activation (ISI) across trials: {np.mean(std_activation):.4f}")

    activations_sspa = [res['activation_sspa'] for res in results if res['activation_sspa'] is not None]
    if len(activations_sspa) > 1:
        activations_sspa = np.vstack(activations_sspa)
        std_activation_sspa = np.std(activations_sspa, axis=0, ddof=1)
        print(f"Mean SD of muscle activation (SSPA) across trials: {np.mean(std_activation_sspa):.4f}")

    # 2D trajectory: SD along the normal to the mean PE-SE curve, then averaged (see exp13)
    all_xy = np.stack([np.column_stack((res['pe'], res['se'])) for res in results], axis=0)   # (n_trials, MOVEMENT_SAMPLES, 2)
    if all_xy.shape[0] > 1:
        mean_xy = np.mean(all_xy, axis=0)                       # (MOVEMENT_SAMPLES, 2)
        dxy = np.gradient(mean_xy, axis=0)                      # tangent of the mean curve
        tangent_norm = np.linalg.norm(dxy, axis=1, keepdims=True)
        tangent_norm[tangent_norm == 0] = 1.0
        tangent = dxy / tangent_norm
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))   # unit normal vector
        diff = all_xy - mean_xy[None, :, :]                     # (n_trials, MOVEMENT_SAMPLES, 2)
        normal_dist = np.sum(diff * normal[None, :, :], axis=2) # signed deviation along the normal
        std_normal = np.std(normal_dist, axis=0, ddof=1)        # (MOVEMENT_SAMPLES,)
        print(f"Mean SD of 2D trajectory (normal to mean curve): {np.mean(std_normal):.2f} deg")

    # apply the plot-only selection defined above (the SD analysis above used all pairs)
    results_to_plot = results if selected is None else [results[i] for i in selected]

    # one distinct color per trial
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # overlaid muscle activations (all trials, resampled to a common grid)
    if PLOT_MUSCLE_ACTIVATION:
        fig, ax = plt.subplots(figsize=(16, 4))
        for i, res in enumerate(results_to_plot):
            c = colors[i % 10]
            if res['activation_isi'] is not None:
                ax.plot(res['movement_pct'], res['activation_isi'],
                        color=c, linestyle='solid', label=f"trial {res['trial']} (ISI)")
            if res['activation_sspa'] is not None:
                ax.plot(res['movement_pct'], res['activation_sspa'],
                        color=c, linestyle='dashed', label=f"trial {res['trial']} (SSPA)")
        ax.set_xlabel('Movement [%]')
        ax.set_ylabel('activation')
        ax.set_title('Muscle activation - all trials')
        ax.set_xticks(np.arange(0, 101, 10))
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        ax.legend()

    # overlaid PE/SE degrees of freedom (all trials, resampled to a common grid)
    if PLOT_2D_TRAJECTORY:
        fig, ax = plt.subplots(figsize=(16, 4))
        for i, res in enumerate(results_to_plot):
            c = colors[i % 10]
            ax.plot(res['movement_pct'], res['pe'], color=c, linestyle='solid')
            ax.plot(res['movement_pct'], res['se'], color=c, linestyle='dashed')
        legend_items = [Line2D([0], [0], color=colors[i % 10], lw=2.5, label=f"trial {res['trial']}")
                        for i, res in enumerate(results_to_plot)]
        legend_items += [
            Line2D([0], [0], color='black', lw=2.5, linestyle='solid', label='PE'),
            Line2D([0], [0], color='black', lw=2.5, linestyle='dashed', label='SE'),
        ]
        ax.set_xlabel('Movement [%]')
        ax.set_ylabel('DoF [°]')
        ax.set_title('DoFs - all trials')
        ax.set_xticks(np.arange(0, 101, 10))
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        ax.legend(handles=legend_items)

    # show all the produced figures at once
    plt.show(block=True)


if __name__ == '__main__':
    main()