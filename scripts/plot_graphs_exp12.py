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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

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

def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_12')

    # define initial and final times in percentage of the overall duration (to be refined for each trial)
    time_0_percent = 0
    time_final_percent = 99

    # select the appropriate bag file depending on the subject and trial considered
    subject = 9 # available 1, 2, 3, 4, 5, 6, 7, 8
    trial = 6   # available 11-8 (subject 1), 1-8 (subject 2, but reps 1,5,6 did not go well), 1-5 (subject 3), 1-6 (subject 4), 
                # 1-6 (subject 5), 1-6 (subject 6), 1-8 (subject 7), 1-6 (subject 8), 1-6 (subject 9).
    if subject == 1:
        if trial==1:
            bag_file_name = 'P1_active_1.bag'
        elif trial==2:
            bag_file_name = 'P1_active_2.bag'
            time_0_percent = 52
            time_final_percent = 68 # one way
            # time_final_percent = 90 # full
        elif trial==3:
            bag_file_name = 'P1_active_3.bag'
            time_0_percent = 22
            time_final_percent = 37 # one way
            # time_final_percent = 52 # full
        elif trial==4:
            bag_file_name = 'P1_active_4.bag'
        elif trial==5:
            bag_file_name = 'P1_active_5.bag'
            time_0_percent = 20
            time_final_percent = 41 # one way
            # time_final_percent = 52 # full
        elif trial==6:
            bag_file_name = 'P1_active_6.bag'
            time_0_percent = 62
            time_final_percent = 76 # one way
            # time_final_percent = 90 # full
        elif trial==7:
            bag_file_name = 'P1_active_7.bag'
        elif trial==8:
            bag_file_name = 'P1_active_8.bag'
            time_0_percent = 15
            time_final_percent = 29 # one way
            # time_final_percent = 43 # full
    elif subject==2:
        if trial==1:
            print("this rep did not go well, be careful when analyzing the results!")
            bag_file_name = 'P2_active_1.bag'
        elif trial==2:
            bag_file_name = 'P2_active_2.bag'
            time_0_percent = 50
            time_final_percent = 70 # one way
            # time_final_percent = 89 # full
        elif trial==3:
            print("this rep did not go well, be careful when analyzing the results!")
            bag_file_name = 'P2_active_3.bag'
        elif trial==4:
            bag_file_name = 'P2_active_4.bag'
        elif trial==5:
            print("this rep did not go well, be careful when analyzing the results!")
            bag_file_name = 'P2_active_5.bag'
            time_0_percent = 13
            time_final_percent = 38 # one way
            # time_final_percent = 51 # full
        elif trial==6:
            print("this rep did not go well, be careful when analyzing the results!")
            bag_file_name = 'P2_active_6.bag'
        elif trial==7:
            bag_file_name = 'P2_active_7.bag'
            time_0_percent = 5
            time_final_percent = 21 # one way
            # time_final_percent = 45 # full
        elif trial==8:
            time_0_percent = 15
            time_final_percent = 32 # one way
            # time_final_percent = 50 # full
            bag_file_name = 'P2_active_8.bag'
    elif subject==3:
        if trial==1:
            bag_file_name = 'P3_active_1.bag'
            time_0_percent = 8
            time_final_percent = 25 # one way
            # time_final_percent = 50 # full
        elif trial==2:
            bag_file_name = 'P3_active_2.bag'
            time_0_percent = 63
            time_final_percent = 75 # one way
            # time_final_percent = 93 # full
        elif trial==3:
            bag_file_name = 'P3_active_3.bag'
        elif trial==4:
            bag_file_name = 'P3_active_4.bag'
            time_0_percent = 7
            time_final_percent = 21 # one way
            # time_final_percent = 60 # full
        elif trial==5:
            bag_file_name = 'P3_active_5.bag'
            time_0_percent = 10
            time_final_percent = 26 # one way
            # time_final_percent = 50 # full
    elif subject==4:
        if trial==1:
            bag_file_name = 'P4_active_1.bag'
            time_0_percent = 18
            time_final_percent = 33 # one way
            # time_final_percent = 55 # full
        elif trial==2:
            bag_file_name = 'P4_active_2.bag'
            time_0_percent = 18
            time_final_percent = 35 # one way
            # time_final_percent = 53 # full
        elif trial==3:
            bag_file_name = 'P4_active_3.bag'
        elif trial==4:
            bag_file_name = 'P4_active_4.bag'
            time_0_percent = 59
            time_final_percent = 75 # one way
            # time_final_percent = 90 # full
        elif trial==5:
            bag_file_name = 'P4_active_5.bag'
        elif trial==6:
            bag_file_name = 'P4_active_6.bag'
    elif subject==5:
        if trial==1:
            bag_file_name = 'P5_active_1.bag'
            time_0_percent = 10
            time_final_percent = 28 # one way
            # time_final_percent = 47 # full
        elif trial==2:
            bag_file_name = 'P5_active_2.bag'
        elif trial==3:
            bag_file_name = 'P5_active_3.bag'
            time_0_percent = 54
            time_final_percent = 77 # one way
            # time_final_percent = 92 # full
        elif trial==4:
            bag_file_name = 'P5_active_4.bag'
            time_0_percent = 54
            time_final_percent = 84 # one way
            # time_final_percent = 94 # full
        elif trial==5:
            bag_file_name = 'P5_active_5.bag'
            time_0_percent = 54
            time_final_percent = 76 # one way
            # time_final_percent = 94 # full
        elif trial==6:
            bag_file_name = 'P5_active_6.bag'
            time_0_percent = 18
            time_final_percent = 33 # one way
            # time_final_percent = 58 # full
    elif subject==6:
        if trial==1:
            bag_file_name = 'P6_active_1.bag'
            time_0_percent = 60
            time_final_percent = 88 # one way
            # time_final_percent = 97 # full
        elif trial==2:
            bag_file_name = 'P6_active_2.bag'
            time_0_percent =13
            time_final_percent = 44 # one way
            # time_final_percent = 58 # full
        elif trial==3:
            bag_file_name = 'P6_active_3.bag'
        elif trial==4:
            bag_file_name = 'P6_active_4.bag'
            time_0_percent =15
            time_final_percent = 43 # one way
            # time_final_percent = 54 # full
        elif trial==5:
            bag_file_name = 'P6_active_5.bag'
            time_0_percent =54
            time_final_percent = 83 # one way
            # time_final_percent = 91 # full
        elif trial==6:
            bag_file_name = 'P6_active_6.bag'
            time_0_percent = 62
            time_final_percent = 81 # one way
            # time_final_percent = 93 # full
    elif subject==7:
        if trial==1:
            bag_file_name = 'P7_active_1.bag'
        elif trial==2:
            bag_file_name = 'P7_active_2.bag'
        elif trial==3:
            bag_file_name = 'P7_active_3.bag'
            time_0_percent = 12
            time_final_percent = 32 # one way
            # time_final_percent = 50 # full
        elif trial==4:
            bag_file_name = 'P7_active_4.bag'
            time_0_percent = 60
            time_final_percent = 78 # one way
            # time_final_percent = 94 # full
        elif trial==5:
            bag_file_name = 'P7_active_5.bag'
            time_0_percent = 21
            time_final_percent = 33 # one way
            # time_final_percent = 45 # full
        elif trial==6:
            bag_file_name = 'P7_active_6.bag'
        elif trial==7:
            bag_file_name = 'P7_active_7.bag'
            time_0_percent = 18
            time_final_percent = 31 # one way
            # time_final_percent = 47 # full
        elif trial==8:
            bag_file_name = 'P7_active_8.bag'
            time_0_percent = 18
            time_final_percent = 31 # one way
            # time_final_percent = 49 # full 
    elif subject==8:
        if trial==1:
            bag_file_name = 'P8_active_1.bag'
        elif trial==2:
            bag_file_name = 'P8_active_2.bag'
            time_0_percent = 10
            time_final_percent = 25 # one way
            # time_final_percent = 49 # full
        elif trial==3:
            bag_file_name = 'P8_active_3.bag'
        elif trial==4:
            bag_file_name = 'P8_active_4.bag'
            time_0_percent = 15
            time_final_percent = 28 # one way
            # time_final_percent = 47 # full
        elif trial==5:
            bag_file_name = 'P8_active_5.bag'
            time_0_percent = 42
            time_final_percent = 61 # one way
            # time_final_percent = 93 # full
        elif trial==6:
            bag_file_name = 'P8_active_6.bag'
            time_0_percent = 12.5
            time_final_percent = 30 # one way
            # time_final_percent = 43 # full
    elif subject==9:
        if trial==1:
            bag_file_name = 'P9_active_1.bag'
            time_0_percent = 13
            time_final_percent = 34 # one way
            # time_final_percent = 54 # full
        elif trial==2:
            bag_file_name = 'P9_active_2.bag'
            time_0_percent = 58
            time_final_percent = 77 # one way
            # time_final_percent = 95 # full
        elif trial==3:
            bag_file_name = 'P9_active_3.bag'
            time_0_percent = 10
            time_final_percent = 23 # one way
            # time_final_percent = 50 # full
        elif trial==4:
            bag_file_name = 'P9_active_4.bag'
            time_0_percent = 15
            time_final_percent = 31 # one way
            # time_final_percent = 50 # full
        elif trial==5:
            bag_file_name = 'P9_active_5.bag'
            time_0_percent = 14
            time_final_percent = 33 # one way
            # time_final_percent = 55 # full
        elif trial==6:
            bag_file_name = 'P9_active_6.bag'
            time_0_percent = 54
            time_final_percent = 75 # one way
            # time_final_percent = 92 # full

    # load the strainmap dictionary used in the experiment
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models', 'Strain Maps', 'Active', 'differentiable_strainmaps_SSPA.pkl')

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
                optimal_strain = np.hstack((data[8, 0], time_curr))
                optimal_trajectory = np.hstack((data[0:6,:], time_curr * np.ones((6,1))))
            else:
                optimal_strain = np.vstack((optimal_strain, np.hstack((data[8, 0], time_curr))))
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
    optimal_strain[:,-1] = optimal_strain[:,-1] - init_time
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

    optimal_strain = optimal_strain[(optimal_strain[:,-1]>0) & (optimal_strain[:,-1]<end_time)]
    filtered_trajectories = []                      # also filter the optimal trajectories
    for i in range(optimal_trajectory.shape[0]):
        slice_i = optimal_trajectory[i, :, :]
        filtered_slice = slice_i[(slice_i[:, -1] > 0) & (slice_i[:, -1] < end_time)]
        filtered_trajectories.append(filtered_slice)

    optimal_trajectory = np.stack(filtered_trajectories, axis=0)        # stack everything into a 3D array again

    # we now instantiate the strainmap
    desired_activation = 0
    strainmap = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[100, 4], desired_activation)

    # we want to extract the time axes from both the shoulder trajectory and the muscle activation estimates
    t_shoulder = estimated_shoulder_state[:, -1]
    t_muscle   = muscle_activation_selected[:, -1] if muscle_activation_selected is not None else None
    # share time normalization
    t_min = min(t_shoulder.min(), t_muscle.min() if t_muscle is not None else t_shoulder.min())
    t_max = max(t_shoulder.max(), t_muscle.max() if t_muscle is not None else t_shoulder.max())
    norm = Normalize(vmin=t_min, vmax=t_max)

    # selecting white -> dark blue colormap ---
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "darkblue"])
    

    # visualize the actual BATON trajectories on the strainmap corresponding to the selected level of activation
    start = np.array([60, 60])
    goal_1 = np.array([45, 95])
    fig = plt.figure()
    ax = fig.add_subplot()
    cb = ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=strainmap.min(), vmax=strainmap.max())
    ax.plot(np.rad2deg(estimated_shoulder_state[:, 0]), np.rad2deg(estimated_shoulder_state[:, 2]))
    ax.scatter(start[0], start[1], label = 'goal', c = 'green', edgecolors='black')
    ax.scatter(goal_1[0], goal_1[1], label = 'start', c='red', edgecolors='black')
    ax.set_ylim(50, 110)
    ax.set_xlim(25, 100)
    ax.set_xlabel("Plane of Elevation [°]")
    ax.set_ylabel("Shoulder Elevation [°]")
    ax.legend()
    ax.set_title("EXP12: Trajectory on strainmap (0 activation)")
    fig.colorbar(cb)

    # here we want to evaluate the actual strain that subjects were experiencing during the experiment, 
    # by looking at the optimal strain trajectory that we computed online during the experiment and that is stored in the bag file.
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(optimal_strain[:, -1], optimal_strain[:, 0])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Optimal strain')
    ax.set_title('EXP12: Optimal strain trajectory during the experiment')

    # display information regarding maximum and average strain of the experiment, together with participant ID and trial number
    print('subject: ', subject)
    print('trial: ', trial)
    print('    Maximum optimal strain during the experiment: ', np.max(optimal_strain[:, 0]))
    print('    Average optimal strain during the experiment: ', np.mean(optimal_strain[:, 0]))

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
    fig.suptitle("EXP12: individual shoulder angles (with human)")

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
    fig.suptitle("EXP12: EE cartesian position")

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
    fig.suptitle("EXP12: EE orientation")

    # visualize magnitude of interaction force
    if interaction_torque_x is not None:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(interaction_torque_x[:,-1], interaction_torque_x[:,0])
        ax.set_ylabel('[N]')
        ax.set_title('Interaction torque (x)')

    if muscle_activation_selected is not None:
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(muscle_activation_selected[:,-1], muscle_activation_selected[:,0], label='ISI')
        ax.legend()
        ax.set_ylabel('activation')
        ax.set_title('Muscle activation')

        # print to screen the average and maximum muscle activation for the selected muscle
        print('    Average activation: ', np.mean(muscle_activation_selected[:,0]))
        print('    Maximum activation: ', np.max(muscle_activation_selected[:,0]))

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

        ax.set_xlabel("Plane of Elevation [°]")
        ax.set_ylabel("Shoulder Elevation [°]")

        # Activation colorbar
        if with_cmap:
            sm = plt.cm.ScalarMappable(cmap=red_cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Muscle activation")

        # Custom legend
        ax.legend(handles=legend_items)
    
    plt.show(block=True)


if __name__ == '__main__':
    main()