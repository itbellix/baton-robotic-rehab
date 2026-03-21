"""
Script to analyze the performance of BATON on multi-subject trial, comparing it to strain values that subject could have
encountered if BATON did not account for their experimental activations and AR values.
This script produces the results that I aggregated in Table 2 of the paper.
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
import re

num_params = 6
def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    '''
    Function used for the 2D interpolation
    '''
    return amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2)+(y-y0)**2/(2*sigma_y**2)))+offset

def generate_approximated_strainmap(file_strainmaps, ar_value, act_value = 0):
    """
    Generate the approximated strainmap for a given AR and activation value.
    Returns:
        fit           : 2D strainmap
        pe_datapoints : PE axis in degrees
        se_datapoints : SE axis in degrees
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
        # print("remember to check the strainmap that you want to visualize!")
        params_gaussians = strainmaps_dict['all_params_gaussians'][index_strainmap_current][act_index]
        num_gaussians = strainmaps_dict['num_gaussians'][index_strainmap_current][act_index]

    fit = np.zeros((pe_datapoints.shape[0], se_datapoints.shape[0]))
    for i in range(len(params_gaussians)//num_params):
        fit += gaussian_2d(X_norm, Y_norm, *params_gaussians[i*num_params:i*num_params+num_params])

    return fit, pe_datapoints, se_datapoints

def project_trajectory_onto_strainmap(estimated_shoulder_state, strainmap, pe_datapoints, se_datapoints):
    """
    Project shoulder trajectory onto the strainmap grid.

    Parameters
    ----------
    estimated_shoulder_state : np.ndarray
        NxM array where:
        - column 0 = PE in radians
        - column 2 = SE in radians
    strainmap : np.ndarray
        2D array with shape (len(pe_datapoints), len(se_datapoints))
    pe_datapoints : np.ndarray
        1D array of PE grid values in degrees
    se_datapoints : np.ndarray
        1D array of SE grid values in degrees

    Returns
    -------
    pe_traj_deg : np.ndarray
        PE trajectory in degrees
    se_traj_deg : np.ndarray
        SE trajectory in degrees
    pe_idx : np.ndarray
        Nearest PE grid indices
    se_idx : np.ndarray
        Nearest SE grid indices
    strain_along_traj : np.ndarray
        Strain value sampled from strainmap at each trajectory point
    """

    pe_traj_deg = np.rad2deg(estimated_shoulder_state[:, 0])
    se_traj_deg = np.rad2deg(estimated_shoulder_state[:, 2])

    # clip to strainmap bounds
    pe_traj_deg = np.clip(pe_traj_deg, pe_datapoints[0], pe_datapoints[-1])
    se_traj_deg = np.clip(se_traj_deg, se_datapoints[0], se_datapoints[-1])

    # nearest-neighbor projection onto grid
    pe_idx = np.abs(pe_datapoints[:, None] - pe_traj_deg[None, :]).argmin(axis=0)
    se_idx = np.abs(se_datapoints[:, None] - se_traj_deg[None, :]).argmin(axis=0)

    strain_along_traj = strainmap[pe_idx, se_idx]

    return pe_traj_deg, se_traj_deg, pe_idx, se_idx, strain_along_traj


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

def load_experiment_summary(results_folder,
                            participants_to_consider=None,
                            trials_to_consider=None):
    """
    Load multisubject experiment results from .npz files.

    Expected file naming convention:
        exp12_sX_tY.npz
    where:
        X = participant number
        Y = trial number

    Each file must contain:
        - trajectory
        - optimal_strain
        - muscle_activation

    Returns
    -------
    results : dict
        Nested dictionary such that:
        results[participant][trial]["avg_activation"]
        results[participant][trial]["max_activation"]
        results[participant][trial]["min_activation"]
        results[participant][trial]["avg_strain"]
        results[participant][trial]["max_strain"]
        results[participant][trial]["min_strain"]
        results[participant][trial]["trajectory"]

    Raises
    ------
    ValueError
        If requested participants/trials are not provided consistently.
    FileNotFoundError
        If a requested .npz file is missing.
    KeyError
        If a required variable is missing inside a .npz file.
    """

    if participants_to_consider is None:
        raise ValueError("participants_to_consider must be provided.")

    if trials_to_consider is None:
        raise ValueError("trials_to_consider must be provided.")

    results = {}

    for participant in participants_to_consider:
        if participant not in trials_to_consider:
            raise ValueError(
                f"Participant {participant} is listed in participants_to_consider "
                f"but has no entry in trials_to_consider."
            )

        results[participant] = {}

        for trial in trials_to_consider[participant]:
            filename = f"exp12_s{participant}_t{trial}.npz"
            filepath = os.path.join(results_folder, filename)

            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    f"Required file not found for participant {participant}, "
                    f"trial {trial}: {filepath}"
                )

            data = np.load(filepath, allow_pickle=True)

            required_keys = ["trajectory", "optimal_strain", "muscle_activation"]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise KeyError(
                    f"File {filepath} is missing required keys: {missing_keys}"
                )

            trajectory = data["trajectory"]
            optimal_strain = np.asarray(data["optimal_strain"]).squeeze()
            muscle_activation = np.asarray(data["muscle_activation"]).squeeze()

            # Safety checks
            if optimal_strain.size == 0:
                raise ValueError(f"'optimal_strain' is empty in file {filepath}")
            if muscle_activation.size == 0:
                raise ValueError(f"'muscle_activation' is empty in file {filepath}")

            results[participant][trial] = {
                "avg_activation": float(np.mean(muscle_activation[:,0])),   # muscle activation is first column, time is the second one
                "max_activation": float(np.max(muscle_activation[:,0])),
                "min_activation": float(np.min(muscle_activation[:,0])),
                "avg_strain": float(np.mean(optimal_strain[:,0])),   # optimal strain is first column, time is the second one
                "max_strain": float(np.max(optimal_strain[:,0])),
                "min_strain": float(np.min(optimal_strain[:,0])),
                "trajectory": trajectory
            }

    return results


def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_13')

    visualize_projected_trajectories_on_straimaps = False

    # define initial and final times in percentage of the overall duration
    time_0_percent = 0
    time_final_percent = 99

    bag_file_name = 'BATON_simulated_trajectories_0_activation.bag' # containing BATON trajectories with 0 activation

    multi_participants_results_path = os.path.join(path_to_repo, 'Personal_Results', 'exp12_analysis')

    participants_to_consider = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    trials_to_consider = {
        1: [5, 6, 8],       # trial 7 had problems
        2: [5, 7, 8],       # trial 6 had problems
        3: [1, 4, 5],
        4: [1, 2, 4],
        5: [4, 5, 6],
        6: [4, 5, 6],
        7: [5, 7 ,8],
        8: [4, 5, 6],
        9: [4, 5, 6],
        10: [4, 2, 6]
    }

    short_rep_flag = True
    
    # load the strainmap dictionary used in the experiment
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models', 'Strain Maps', 'Active', 'differentiable_strainmaps_SSPA.pkl')

    # load the results for the multisubject experiment, to be used to select the right strainmaps and evaluate BATON's 0-activation trajectory
    # The results are stored in a .npz file, named "exp12_sX_tY.npz", where X is the subject number and Y is the trial number. Each file contains:
    # - trajectory
    # - optimal_strain
    # - muscle_activation
    results_multisubject_experiment = load_experiment_summary(multi_participants_results_path,
                                                              participants_to_consider=participants_to_consider,
                                                              trials_to_consider=trials_to_consider)

    # instantiate variables (they will be Mx4 matrices, where M is variable -number of data- and the last column is the timestamp)
    estimated_shoulder_state = None
    xyz_cmd = None
    xyz_curr = None

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

        # Now, let's filter the data to retain only the interesting part of the experiment
    # (i.e., when the subject is wearing the brace properly and the robot is moving)
    init_time = xyz_curr[int(xyz_curr.shape[0]/100*time_0_percent), -1]         # identify initial timestep
    end_time = xyz_curr[int(xyz_curr.shape[0]/100*time_final_percent), -1] - init_time # identify final timestep

    estimated_shoulder_state[:,-1] = estimated_shoulder_state[:,-1] - init_time   # center time values starting at initial time
    xyz_curr[:,-1] = xyz_curr[:,-1] - init_time
    xyz_cmd[:,-1] = xyz_cmd[:,-1] - init_time

    estimated_shoulder_state = estimated_shoulder_state[(estimated_shoulder_state[:,-1]>0) & (estimated_shoulder_state[:,-1]<end_time)]
    xyz_curr = xyz_curr[(xyz_curr[:,-1]>0) & (xyz_curr[:,-1]<end_time)]    # retain data after initial time
    xyz_cmd = xyz_cmd[(xyz_cmd[:,-1]>0) & (xyz_cmd[:,-1]<end_time)]

    # initialize results dictionary for the simulation
    results_multisubject_simulation = {}

    # now we loop over all of the trials of the multisubject experiment, to select the right strainmap and evaluate the 0-activation trajectory
    for participant in participants_to_consider:
        results_multisubject_simulation[participant] = {}   # allocate cell for participant in results dictionary

        for trial in trials_to_consider[participant]:
            ar_min = results_multisubject_experiment[participant][trial]["trajectory"][:, 4].min()
            ar_max = results_multisubject_experiment[participant][trial]["trajectory"][:, 4].max()
            avg_ar = results_multisubject_experiment[participant][trial]["trajectory"][:, 4].mean()

            # load the strainmap corresponding to the average and maximum activation of the participant's trial
            strainmap_avg, pe_datapoints, se_datapoints = generate_approximated_strainmap(file_strainmaps, 
                                                                                          avg_ar, 
                                                                                          results_multisubject_experiment[participant][trial]["avg_activation"])
            strainmap_maxAct_minAR, _, _ = generate_approximated_strainmap(file_strainmaps,
                                                                           ar_min, 
                                                                           results_multisubject_experiment[participant][trial]["max_activation"])
            
            strainmap_maxAct_maxAR, _, _ = generate_approximated_strainmap(file_strainmaps,
                                                                           ar_max, 
                                                                           results_multisubject_experiment[participant][trial]["max_activation"])

            # now we can evaluate the 0-activation trajectory on the TWO selected strainmap, to see how it performs in terms of strain values
            # (this will be done in the paper to show that the 0-activation trajectory is not good in terms of optimal strain, and that the strainmap can be used to select better trajectories for the patient)
            pe_traj_deg, se_traj_deg, pe_idx, se_idx, strain_along_traj_avg = project_trajectory_onto_strainmap(estimated_shoulder_state,
                                                                                                                strainmap_avg,
                                                                                                                pe_datapoints,
                                                                                                                se_datapoints)
            
            pe_traj_deg_max, se_traj_deg_max, pe_idx_max, se_idx_max, strain_along_traj_extreme1 = project_trajectory_onto_strainmap(
                estimated_shoulder_state,
                strainmap_maxAct_minAR,
                pe_datapoints,
                se_datapoints)
            
            pe_traj_deg_max, se_traj_deg_max, pe_idx_max, se_idx_max, strain_along_traj_extreme2 = project_trajectory_onto_strainmap(
                estimated_shoulder_state,
                strainmap_maxAct_maxAR,
                pe_datapoints,
                se_datapoints)
            
            # compute summary values
            avg_strain_on_avg_act = np.round(np.mean(strain_along_traj_avg), 2)
            max_strain_on_avg_act = np.round(np.max(strain_along_traj_avg), 2)

            # avg_strain_on_max_act = np.round(np.mean(strain_along_traj_max), 2)
            max_strain_on_max_act1 = np.round(np.max(strain_along_traj_extreme1), 2)
            max_strain_on_max_act2 = np.round(np.max(strain_along_traj_extreme2), 2)
            max_strain_on_max_act = max(max_strain_on_max_act1, max_strain_on_max_act2)

            # compute also minumum strain of the trajectory, in the worst case scenario of maximum activation and extreme values of AR
            min_strain_on_max_act1 = np.round(np.min(strain_along_traj_extreme1), 2)
            min_strain_on_max_act2 = np.round(np.min(strain_along_traj_extreme2), 2)
            min_strain_on_max_act = min(min_strain_on_max_act1, min_strain_on_max_act2)
            
            # store results with same key structure
            results_multisubject_simulation[participant][trial] = {
                "avg_activation_strainmap": {
                    "avg_strain": avg_strain_on_avg_act,
                    "max_strain": max_strain_on_avg_act,},
                "max_activation_strainmap": {
                    "avg_strain": None,
                    "max_strain": max_strain_on_max_act,
                    "min_strain": min_strain_on_max_act
                },}

            if visualize_projected_trajectories_on_straimaps:
                fig, ax = plt.subplots(figsize=(8, 6))
                cb = ax.imshow(strainmap_avg.T, 
                            origin='lower', 
                            cmap='hot', 
                            extent=[pe_datapoints[0], pe_datapoints[-1], se_datapoints[0], se_datapoints[-1]], 
                            vmin=strainmap_avg.min(), 
                            vmax=strainmap_avg.max())
                ax.plot(pe_traj_deg, se_traj_deg, linewidth=2, label='Trajectory')
                ax.set_xlabel('Plane of elevation [deg]')
                ax.set_ylabel('Shoulder elevation [deg]')
                ax.set_title(f'Participant {participant}, Trial {trial}, avg activation {results_multisubject_experiment[participant][trial]["avg_activation"]:.2f}')
                ax.legend()
                plt.colorbar(cb, ax=ax, label='Baseline strain')

                fig, ax = plt.subplots(figsize=(8, 6))
                cb = ax.imshow(strainmap_maxAct_maxAR.T, 
                            origin='lower', 
                            cmap='hot', 
                            extent=[pe_datapoints[0], pe_datapoints[-1], se_datapoints[0], se_datapoints[-1]], 
                            vmin=strainmap_maxAct_maxAR.min(), 
                            vmax=strainmap_maxAct_maxAR.max())
                ax.plot(pe_traj_deg, se_traj_deg, linewidth=2, label='Trajectory')
                ax.set_xlabel('Plane of elevation [deg]')
                ax.set_ylabel('Shoulder elevation [deg]')
                ax.set_title(f'Participant {participant}, Trial {trial}, max activation {results_multisubject_experiment[participant][trial]["max_activation"]:.2f}')
                ax.legend()
                plt.colorbar(cb, ax=ax, label='Baseline strain')

                fig, ax = plt.subplots(figsize=(8, 6))
                cb = ax.imshow(strainmap_maxAct_minAR.T, 
                            origin='lower', 
                            cmap='hot', 
                            extent=[pe_datapoints[0], pe_datapoints[-1], se_datapoints[0], se_datapoints[-1]], 
                            vmin=strainmap_maxAct_minAR.min(), 
                            vmax=strainmap_maxAct_minAR.max())
                ax.plot(pe_traj_deg, se_traj_deg, linewidth=2, label='Trajectory')
                ax.set_xlabel('Plane of elevation [deg]')
                ax.set_ylabel('Shoulder elevation [deg]')
                ax.set_title(f'Participant {participant}, Trial {trial}, max activation {results_multisubject_experiment[participant][trial]["max_activation"]:.2f}')
                ax.legend()
                plt.colorbar(cb, ax=ax, label='Baseline strain')

    if visualize_projected_trajectories_on_straimaps:        
        plt.show()

    # at the very end, we subtract the reults from the experiment from the results from the simulation ,to evaluate strain reduction with BATON's adaptability.
    # In particular, we print for every participand and trial the difference between:
    # simulated average strain - experimental average strain on average strainmap
    # simulated maximum strain - experimental maximum strain on maximum strainmap

    for participant in participants_to_consider:
        for trial in trials_to_consider[participant]:
            exp_avg_strain = results_multisubject_experiment[participant][trial]["avg_strain"]
            exp_max_strain = results_multisubject_experiment[participant][trial]["max_strain"]
            exp_min_strain = results_multisubject_experiment[participant][trial]["min_strain"]

            sim_avg_strain = results_multisubject_simulation[participant][trial]["avg_activation_strainmap"]["avg_strain"]
            sim_max_strain = results_multisubject_simulation[participant][trial]["max_activation_strainmap"]["max_strain"]
            sim_min_strain = results_multisubject_simulation[participant][trial]["max_activation_strainmap"]["min_strain"]

            avg_strain_reduction = np.round(sim_avg_strain - exp_avg_strain, 2)
            max_strain_reduction = np.round(sim_max_strain - exp_max_strain, 2)
            min_strain_reduction = np.round(sim_min_strain - exp_min_strain, 2)

            print(f"Participant {participant}, Trial {trial}:")
            print(f"  Average strain reduction: {avg_strain_reduction}")
            print(f"  Maximum strain reduction: {max_strain_reduction}")
            print(f"  Minimum strain reduction: {min_strain_reduction}")

    # now we want to compute for each participant and print to screen:
    # - the mean across trials of the experimental average strain plus/minus standard deviation
    # - the mean across trials of the experimental maximum strain plus/minus standard deviation
    # - the mean across trials of the exprimental minimum strain plus/minus standard deviation
    # - the mean across trials of the simulated average strain on the average activation strainmap plus/minus standard deviation
    # - the mean across trials of the simulated maximum strain on the maximum activation strainmap plus/minus standard deviation
    # - the mean across trials of the simulated minimum strain on the maximum activation strainmap plus/minus standard deviation
    # - reduction in average strain: mean simulated average strain on average activation strainmap - mean experimental average strain
    # - reduction in maximum strain: mean simulated maximum strain on maximum activation strainmap - mean experimental maximum strain
    # - reduction in minimum strain: mean simulated minimum strain on maximum activation strainmap - mean experimental minimum strain
    # ----- containers for group-level statistics -----
    group_exp_avg_means = []
    group_sim_avg_means = []
    group_reduction_avg = []

    group_exp_max_means = []
    group_sim_max_means = []
    group_reduction_max = []

    group_exp_min_means = []
    group_sim_min_means = []
    group_reduction_min = []

    for participant in participants_to_consider:
        exp_avg_strains = []
        exp_max_strains = []
        exp_min_strains = []
        sim_avg_strains = []
        sim_max_strains = []
        sim_min_strains = []

        for trial in trials_to_consider[participant]:
            exp_avg_strains.append(results_multisubject_experiment[participant][trial]["avg_strain"])
            exp_max_strains.append(results_multisubject_experiment[participant][trial]["max_strain"])
            exp_min_strains.append(results_multisubject_experiment[participant][trial]["min_strain"])
            sim_avg_strains.append(results_multisubject_simulation[participant][trial]["avg_activation_strainmap"]["avg_strain"])
            sim_max_strains.append(results_multisubject_simulation[participant][trial]["max_activation_strainmap"]["max_strain"])
            sim_min_strains.append(results_multisubject_simulation[participant][trial]["max_activation_strainmap"]["min_strain"])

        # Skip participants with no trials
        if len(exp_avg_strains) == 0:
            print(f"Participant {participant}: no trials")
            continue

        exp_avg_strains = np.array(exp_avg_strains)
        exp_max_strains = np.array(exp_max_strains)
        exp_min_strains = np.array(exp_min_strains)
        sim_avg_strains = np.array(sim_avg_strains)
        sim_max_strains = np.array(sim_max_strains)
        sim_min_strains = np.array(sim_min_strains)

        # ----- per-participant means -----
        mean_exp_avg_strain = np.mean(exp_avg_strains)
        mean_exp_max_strain = np.mean(exp_max_strains)
        mean_exp_min_strain = np.mean(exp_min_strains)
        mean_sim_avg_strain = np.mean(sim_avg_strains)
        mean_sim_max_strain = np.mean(sim_max_strains)
        mean_sim_min_strain = np.mean(sim_min_strains)

        # ----- per-participant std (across trials) -----
        std_exp_avg_strain = np.std(exp_avg_strains, ddof=1)
        std_exp_max_strain = np.std(exp_max_strains, ddof=1)
        std_exp_min_strain = np.std(exp_min_strains, ddof=1)
        std_sim_avg_strain = np.std(sim_avg_strains, ddof=1)
        std_sim_max_strain = np.std(sim_max_strains, ddof=1)
        std_sim_min_strain = np.std(sim_min_strains, ddof=1)

        # ----- reductions -----
        # ----- percentage reductions relative to simulated means -----
        if mean_sim_avg_strain != 0:
            reduction_avg_percent = 100.0 * (mean_sim_avg_strain - mean_exp_avg_strain) / mean_sim_avg_strain
        else:
            reduction_avg_percent = np.nan

        if mean_sim_max_strain != 0:
            reduction_max_percent = 100.0 * (mean_sim_max_strain - mean_exp_max_strain) / mean_sim_max_strain
        else:
            reduction_max_percent = np.nan

        if mean_sim_min_strain != 0:
            reduction_min_percent = 100.0 * (mean_sim_min_strain - mean_exp_min_strain) / mean_sim_min_strain
        else:
            reduction_min_percent = np.nan

        # ----- store for group statistics -----
        group_exp_avg_means.append(mean_exp_avg_strain)
        group_sim_avg_means.append(mean_sim_avg_strain)
        group_reduction_avg.append(reduction_avg_percent)

        group_exp_max_means.append(mean_exp_max_strain)
        group_sim_max_means.append(mean_sim_max_strain)
        group_reduction_max.append(reduction_max_percent)

        group_exp_min_means.append(mean_exp_min_strain)
        group_sim_min_means.append(mean_sim_min_strain)
        group_reduction_min.append(reduction_min_percent)
        # ----- print per-participant -----
        print(f"Participant {participant}:")
        print(
            f"  Mean experimental average strain: "
            f"{mean_exp_avg_strain:.2f} ± {std_exp_avg_strain:.2f} | "
            f"Mean simulated average strain on avg activation strainmap: "
            f"{mean_sim_avg_strain:.2f} ± {std_sim_avg_strain:.2f} | "
            f"Reduction: {reduction_avg_percent:.2f}%"
        )
        print(
            f"  Mean experimental maximum strain: "
            f"{mean_exp_max_strain:.2f} ± {std_exp_max_strain:.2f} | "
            f"Mean simulated maximum strain on max activation strainmap: "
            f"{mean_sim_max_strain:.2f} ± {std_sim_max_strain:.2f} | "
            f"Reduction: {reduction_max_percent:.2f}%"
        )
        print(
            f"  Mean experimental minimum strain: "
            f"{mean_exp_min_strain:.2f} ± {std_exp_min_strain:.2f} | "
            f"Mean simulated minimum strain on max activation strainmap: "
            f"{mean_sim_min_strain:.2f} ± {std_sim_min_strain:.2f} | "
            f"Reduction: {reduction_min_percent:.2f}%"
        )


    # ============================================================
    #                GROUP-LEVEL RESULTS
    # ============================================================

    if len(group_exp_avg_means) > 0:

        group_exp_avg_means = np.array(group_exp_avg_means)
        group_sim_avg_means = np.array(group_sim_avg_means)
        group_reduction_avg = np.array(group_reduction_avg)

        group_exp_max_means = np.array(group_exp_max_means)
        group_sim_max_means = np.array(group_sim_max_means)
        group_reduction_max = np.array(group_reduction_max)

        group_exp_min_means = np.array(group_exp_min_means)
        group_sim_min_means = np.array(group_sim_min_means)
        group_reduction_min = np.array(group_reduction_min)

        print("\n===== AVERAGE ACROSS PARTICIPANTS =====")

        print(
            f"1. Mean experimental average strain: "
            f"{np.mean(group_exp_avg_means):.2f} ± {np.std(group_exp_avg_means, ddof=1):.2f}"
        )

        print(
            f"2. Mean simulated average strain: "
            f"{np.mean(group_sim_avg_means):.2f} ± {np.std(group_sim_avg_means, ddof=1):.2f}"
        )

        print(
            f"3. Mean reduction (average strain): "
            f"{np.mean(group_reduction_avg):.2f}"
        )

        print(
            f"4. Mean experimental maximum strain: "
            f"{np.mean(group_exp_max_means):.2f} ± {np.std(group_exp_max_means, ddof=1):.2f}"
        )

        print(
            f"5. Mean simulated maximum strain: "
            f"{np.mean(group_sim_max_means):.2f} ± {np.std(group_sim_max_means, ddof=1):.2f}"
        )

        print(
            f"6. Mean reduction (maximum strain): "
            f"{np.mean(group_reduction_max):.2f}%"
        )

        print(
            f"7. Mean experimental minimum strain: "
            f"{np.mean(group_exp_min_means):.2f} ± {np.std(group_exp_min_means, ddof=1):.2f}"
        )

        print(
            f"8. Mean simulated minimum strain: "
            f"{np.mean(group_sim_min_means):.2f} ± {np.std(group_sim_min_means, ddof=1):.2f}"
        )

        print(
            f"9. Mean reduction (minimum strain): "
            f"{np.mean(group_reduction_min):.2f}%"
        )


if __name__ == '__main__':
    main()