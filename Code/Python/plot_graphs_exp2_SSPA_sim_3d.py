"""
Script to analyze the data collected in a rosbag during simulated experiments with varying muscle activations.
The plots produced by this script are used to generate the "layered plot" for the continuous
replannning of the paper (Fig.7 in the paper).
"""

import os
from spatialmath import SO3
from matplotlib.colors import LinearSegmentedColormap
import rosbag
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

num_params = 6
def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    '''
    Function used for the 2D interpolation
    '''
    return amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2)+(y-y0)**2/(2*sigma_y**2)))+offset

def generate_approximated_strainmap(file_strainmaps, ar_value, act_value = 0, pe_lims =np.array([-20, 160]), se_lims = np.array([0, 144])):
    """
    Function to generate the approximated strainmap to consider, given a file containing the
    corresponding dictionary, and the axial rotation value with which to slice it.
    """

    # set the strainmap to operate onto, extracting the information from a file
    with open(file_strainmaps, 'rb') as file:
        strainmaps_dict = pickle.load(file)

    # strainmap span
    max_se = se_lims[1]
    min_se = se_lims[0]

    max_pe = pe_lims[1]
    min_pe = pe_lims[0]

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

    if isinstance(strainmaps_dict['all_params_gaussians'][0], list):
        # varying activation
        print("remember to check the strainmap that you want to visualize!")
        params_gaussians = strainmaps_dict['all_params_gaussians'][index_strainmap_current][act_index]

    fit = np.zeros((pe_datapoints.shape[0], se_datapoints.shape[0]))
    for i in range(len(params_gaussians)//num_params):
        fit += gaussian_2d(X_norm, Y_norm, *params_gaussians[i*num_params:i*num_params+num_params])

    return fit


def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_simulated')

    bag_file_name = '2024-05-14-15-25-19_exp2sim_const0activ.bag'
    title = "Trajectory (0.25->0 activation, ramp)"

    # instantiate variables (they will be Mx4 matrices, where M is variable -number of data- and the last column is the timestamp)
    estimated_shoulder_state = None
    xyz_cmd = None
    xyz_curr = None
    z_uncompensated = None
    angvec_cmd = None
    angvec_curr = None

    # find the strain map parameters
    file_strainmaps = '/home/itbellix/Desktop/GitHub/PTbot_official/Personal_Results/Strains/Active/SSPA_params_strainmaps_num_Gauss_3/SSPA_params_strainmaps_active_3gauss.pkl' 

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
        for _, msg, time_msg in bag.read_messages(topics=['/optimal_cartesian_ref_ee']):
            if xyz_cmd is None:
                timestamps_cmd = time_msg.to_time()
                data_msg = np.reshape(msg.data, (4,4))
                xyz_cmd = np.hstack((data_msg[0:3, 3], timestamps_cmd))
                angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                angvec_cmd = np.hstack((angle, vector, timestamps_cmd))
            else:
                timestamps_cmd = time_msg.to_time()
                data_msg = np.reshape(msg.data, (4,4))
                xyz_cmd = np.vstack((xyz_cmd, np.hstack((data_msg[0:3, 3], timestamps_cmd))))
                angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                angvec_cmd = np.vstack((angvec_cmd, np.hstack((angle, vector, timestamps_cmd))))

        print('Extracting not adjusted ee Z cartesian value')
        for _, msg, time_msg in bag.read_messages(topics=['/uncompensated_z_ref']):
            if z_uncompensated is None:
                timestamps_z = time_msg.to_time()
                z_uncompensated = np.hstack((msg.data, timestamps_z))
            else:
                timestamps_z = time_msg.to_time()
                z_uncompensated = np.vstack((z_uncompensated, np.hstack((msg.data, timestamps_z))))

        # extracting actual ee cartesian poses
        print('Extracting actual ee cartesian poses')
        for _, msg, time_msg in bag.read_messages(topics=['/iiwa7/ee_cartesian_pose']):
            time_curr = time_msg.to_time()
            if time_curr >= xyz_cmd[0, -1]:
                if xyz_curr is None:
                    timestamps_curr = time_msg.to_time()
                    data_msg = np.reshape(msg.pose, (4,4))
                    xyz_curr = np.hstack((data_msg[0:3, 3], timestamps_curr))
                    angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                    angvec_curr = np.hstack((angle, vector, timestamps_curr))
                else:
                    timestamps_curr = time_msg.to_time()
                    data_msg = np.reshape(msg.pose, (4,4))
                    xyz_curr = np.vstack((xyz_curr, np.hstack((data_msg[0:3, 3], timestamps_curr))))
                    angle, vector = SO3(data_msg[0:3, 0:3]).angvec(unit='deg')
                    angvec_curr = np.vstack((angvec_curr, np.hstack((angle, vector, timestamps_curr))))

        print('Extracting optimization outputs')
        optimal_trajectory = None
        optimal_strain = None
        activation_history = None
        for _, msg, time_msg in bag.read_messages(topics=['/optimization_output']):
            data = np.reshape(msg.data[0:-1], (9, 11))
            time_curr = time_msg.to_time()
            if optimal_strain is None:
                optimal_strain = np.hstack((data[8,:], time_curr))
                optimal_trajectory = np.hstack((data[0:6,:], time_curr * np.ones((6,1))))
                activation_history = np.hstack((msg.data[-1], time_curr))
            else:
                optimal_strain = np.vstack((optimal_strain, np.hstack((data[8,:], time_curr))))
                optimal_trajectory = np.vstack((optimal_trajectory, np.hstack((data[0:6,:], time_curr * np.ones((6,1))))))
                activation_history = np.vstack((activation_history, np.hstack((msg.data[-1], time_curr))))

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
    activation_history[:, -1] = activation_history[:,-1] - init_time

    estimated_shoulder_state = estimated_shoulder_state[(estimated_shoulder_state[:,-1]>0) & (estimated_shoulder_state[:,-1]<end_time)]
    xyz_curr = xyz_curr[(xyz_curr[:,-1]>0) & (xyz_curr[:,-1]<end_time)]    # retain data after initial time
    xyz_cmd = xyz_cmd[(xyz_cmd[:,-1]>0) & (xyz_cmd[:,-1]<end_time)]
    if z_uncompensated is not None:
        z_uncompensated = z_uncompensated[(z_uncompensated[:,-1]>0) & (z_uncompensated[:,-1]<end_time)]
    angvec_curr = angvec_curr[(angvec_curr[:,-1]>0) & (angvec_curr[:,-1]<end_time)]
    angvec_cmd = angvec_cmd[(angvec_cmd[:,-1]>0) & (angvec_cmd[:,-1]<end_time)]
    activation_history = activation_history[(activation_history[:,-1]>0) & (activation_history[:,-1]<end_time)]

    filtered_trajectories = []                      # also filter the optimal trajectories
    for i in range(optimal_trajectory.shape[0]):
        slice_i = optimal_trajectory[i, :, :]
        filtered_slice = slice_i[(slice_i[:, -1] > 0) & (slice_i[:, -1] < end_time)]
        filtered_trajectories.append(filtered_slice)

    optimal_trajectory = np.stack(filtered_trajectories, axis=0)        # stack everything into a 3D array again
    
    # select the portion of data to visualize
    len_data = len(estimated_shoulder_state)
    begin_d = int(len_data/100*10)
    end_d = int(len_data/100*100)-1

    begin_time_viz = estimated_shoulder_state[begin_d, -1]
    end_time_viz = estimated_shoulder_state[end_d, -1]

    # retain only activation in this time interval
    activation_history = activation_history[(activation_history[:,-1]>begin_time_viz) & (activation_history[:,-1]<end_time_viz)]
    avg_time_diff = np.mean(np.diff(activation_history[:,1]))
    min_activation = 0
    max_activation = 0.25
    step_activation = 0.025
    activation_on_maps = np.arange(min_activation, max_activation+step_activation, step_activation)

    # set the dimensions of the map that we are interested in
    pe_lims =np.array([40, 65])
    se_lims = np.array([50, 100])
    # pe_lims =np.array([-20, 160])
    # se_lims = np.array([0, 144])

    # first, generate all of the strain maps that we will need, to determine the min and max strain
    min_strain = np.inf
    max_strain = -np.inf
    for index in range(activation_on_maps.shape[0]):
        current_act = activation_on_maps[index]
        strainmap = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[100, 4], act_value=current_act, pe_lims = pe_lims, se_lims = se_lims)
        min_strain = np.min((min_strain, strainmap.min()))
        max_strain = np.max((max_strain, strainmap.max()))

    pe_lims =np.array([-20, 160])
    se_lims = np.array([0, 144])
    
    # now, plot all of the trajectory portions on the correct maps
    for index in range(activation_on_maps.shape[0]):
        current_act = activation_on_maps[index]
        strainmap = generate_approximated_strainmap(file_strainmaps, estimated_shoulder_state[100, 4], act_value=current_act, pe_lims = pe_lims, se_lims = se_lims)
        
        # when are we on the current strain map?
        timestamps_on_map = activation_history[np.abs(activation_history[:, 0] - current_act) <= step_activation/2][:,1]

        if timestamps_on_map.shape[0] > 0:      # check if there is at least one point on the current map
            intervals = []
            current_interval = [timestamps_on_map[0]]

            # Iterate through the timestamps to split into intervals
            for i in range(1, len(timestamps_on_map)):
                if timestamps_on_map[i] - timestamps_on_map[i - 1] > avg_time_diff*5:
                    # if the difference is larger than expected, start a new interval
                    intervals.append(np.array(current_interval))
                    current_interval = [timestamps_on_map[i]]
                else:
                    # Otherwise, continue adding to the current interval
                    current_interval.append(timestamps_on_map[i])

            # Append the last interval
            intervals.append(np.array(current_interval))

            # now, interval the trajectory to retain only the portions that pertain to the current map
            # Initialize a list to hold the filtered rows
            traj_on_map = []

            # Iterate through each interval and filter the data
            for interval in intervals:
                # Filter the estimated_shoulder_state based on the current interval
                row_indexes =  np.where((estimated_shoulder_state[:, -1] >= interval.min()) & (estimated_shoulder_state[:, -1] <= interval.max()))
                filtered_interval = estimated_shoulder_state[row_indexes]
                
                # Append the filtered interval to the list
                traj_on_map.append(filtered_interval)

            # # Combine the filtered intervals into a single array
            # traj_on_map = np.vstack(traj_on_map)

            
            title_fig = title+" act = "+ str(np.round(current_act,3))

            fig = plt.figure()
            ax = fig.add_subplot()
            cb = ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[pe_lims[0], pe_lims[1], se_lims[0], se_lims[1]], vmin=strainmap.min(), vmax=max_strain-0.2)
            for index_segment in range(len(traj_on_map)):
                ax.plot(np.rad2deg(traj_on_map[index_segment][:, 0]), np.rad2deg(traj_on_map[index_segment][:, 2]), color = 'blue')
            
            ax.scatter(np.array([45]), np.array([95]), label = 'goal', c = 'green', edgecolors='black')
            ax.scatter(np.array([60]), np.array([60]), label = 'start', c='red', edgecolors='black')
            ax.set_ylim(50, 100)
            ax.set_xlim(40, 65)
            ax.set_xlabel("Plane of Elevation [°]")
            ax.set_ylabel("Shoulder Elevation [°]")
            ax.legend()
            fig.colorbar(cb)
            ax.set_title(title_fig)
        else:

            title_fig = title+" act = "+ str(np.round(current_act,3))

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[pe_lims[0], pe_lims[1], se_lims[0], se_lims[1]], vmin=strainmap.min(), vmax=max_strain-0.2)
            ax.scatter(np.array([45]), np.array([95]), label = 'goal', c = 'green', edgecolors='black')
            ax.scatter(np.array([60]), np.array([60]), label = 'start', c='red', edgecolors='black')
            ax.set_ylim(50, 100)
            ax.set_xlim(40, 65)
            ax.set_xlabel("Plane of Elevation [°]")
            ax.set_ylabel("Shoulder Elevation [°]")
            ax.legend()
            ax.set_title(title_fig)

    # -----------------------------------------------------------------------------
    plt.show(block=True)


if __name__ == '__main__':
    main()