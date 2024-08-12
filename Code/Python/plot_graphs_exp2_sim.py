"""
Script to explore the response of our trajectory optimization to
sudden changes in the strain map that is considered.
Used to generate results for Fig. 8 in the paper
"""

import os
from bagAnalyzer import *
import pickle
from spatialmath import SO3

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

def generate_strainmap_given_params(params):
    """
    Function to generate a strainmap given the parameters from the user. The inputs are:
    - params: np.array of shape [num_gaussians*6, 1], containing [amplitude, x0, y0, sigma_x, sigma_y, offset]
              of the various gaussians used to approximate the strain landscape.
    """

    # strainmap span
    max_se = 144
    min_se = 0

    max_pe = 160
    min_pe = -20

    step_viz = 0.05

    # create the grids that will be used for interpolation (and visualization)
    pe_datapoints = np.array(np.arange(min_pe, max_pe, step_viz))
    se_datapoints = np.array(np.arange(min_se, max_se, step_viz))

    X,Y = np.meshgrid(pe_datapoints, se_datapoints, indexing='ij')
    X_norm = X/max_pe
    Y_norm = Y/max_se

    fit = np.zeros((pe_datapoints.shape[0], se_datapoints.shape[0]))
    for i in range(len(params)//num_params):
        fit += gaussian_2d(X_norm, Y_norm, *params[i*num_params:i*num_params+num_params])

    return fit



def main():
    # define the required paths
    code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
    path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_simulated')
    # bag_file_name = '2024-03-26-12-50-41_Exp2_noHuman.bag'
    # bag_file_name = '2024-03-26-12-54-22_Exp2_withHuman.bag'
    bag_file_name = '2024-05-06-17-53-38.bag'

    # load the strainmap dictionary used in the experiment
    # file_strainmaps = '/home/itbellix/Desktop/GitHub/PTbot_official/Personal_Results/Strains/Passive/AllMuscles/params_strainmaps_num_Gauss_3/params_strainmaps_num_Gauss_3.pkl' 
    file_strainmaps = '/home/itbellix/Desktop/GitHub/PTbot_official/Personal_Results/Strains/Active/SSPA_params_strainmaps_num_Gauss_3/SSPA_params_strainmaps_active_3gauss.pkl' 

    analyzer = bagAnalyzer(os.path.join(path_to_bag, bag_file_name))
    # analyzer.list_topics_bag(0)

    # instantiate variables (they will be Mx4 matrices, where M is variable -number of data- and the last column is the timestamp)
    estimated_shoulder_state = None
    xyz_cmd = None
    xyz_curr = None
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
    optimal_trajectory_pe = optimal_trajectory[:,:,0]
    optimal_trajectory_se = optimal_trajectory[:,:,2]

    # Now, let's filter the data to retain only the interesting part of the experiment
    # (i.e., when the subject is wearing the brace properly and the robot is moving)
    init_time = xyz_curr[int(xyz_curr.shape[0]/100*5), -1]         # identify initial timestep
    end_time = xyz_curr[int(xyz_curr.shape[0]/100*99), -1] - init_time # identify final timestep

    estimated_shoulder_state[:,-1] = estimated_shoulder_state[:,-1] - init_time   # center time values starting at initial time
    xyz_curr[:,-1] = xyz_curr[:,-1] - init_time
    xyz_cmd[:,-1] = xyz_cmd[:,-1] - init_time
    angvec_curr[:,-1] = angvec_curr[:,-1] - init_time
    angvec_cmd[:,-1] = angvec_cmd[:,-1] - init_time
    optimal_trajectory[:,:,-1] = optimal_trajectory[:,:,-1] - init_time

    estimated_shoulder_state = estimated_shoulder_state[(estimated_shoulder_state[:,-1]>0) & (estimated_shoulder_state[:,-1]<end_time)]
    xyz_curr = xyz_curr[(xyz_curr[:,-1]>0) & (xyz_curr[:,-1]<end_time)]    # retain data after initial time
    xyz_cmd = xyz_cmd[(xyz_cmd[:,-1]>0) & (xyz_cmd[:,-1]<end_time)]
    angvec_curr = angvec_curr[(angvec_curr[:,-1]>0) & (angvec_curr[:,-1]<end_time)]
    angvec_cmd = angvec_cmd[(angvec_cmd[:,-1]>0) & (angvec_cmd[:,-1]<end_time)]

    filtered_trajectories = []                      # also filter the optimal trajectories
    for i in range(optimal_trajectory.shape[0]):
        slice_i = optimal_trajectory[i, :, :]
        filtered_slice = slice_i[(slice_i[:, -1] > 0) & (slice_i[:, -1] < end_time)]
        filtered_trajectories.append(filtered_slice)

    optimal_trajectory = np.stack(filtered_trajectories, axis=0)        # stack everything into a 3D array again

    p1 = np.array([0, 0, 0, 1, 1, 0])
    p2 = np.array([0, 0, 0, 1, 1, 0])
    p3 = np.array([0, 0, 0, 1, 1, 0])
    params_gaussians = np.hstack((p1, p2, p3))
    strainmap_1 = generate_strainmap_given_params(params_gaussians)

    p1 = np.array([3, 72/160, 80/144, 10/160, 5/144, 0])
    p2 = np.array([0, 0, 0, 1, 1, 0])
    p3 = np.array([0, 0, 0, 1, 1, 0])
    params_gaussians = np.hstack((p1, p2, p3))
    strainmap_2 = generate_strainmap_given_params(params_gaussians)

    p1 = np.array([3, 33/160, 67/144, 10/160, 5/144, 0])
    p2 = np.array([0, 0, 0, 1, 1, 0])
    p3 = np.array([0, 0, 0, 1, 1, 0])
    params_gaussians = np.hstack((p1, p2, p3))
    strainmap_3 = generate_strainmap_given_params(params_gaussians)
    

    # visualize 2D trajectory on strainmap (zero activation)
    
    len_data = len(estimated_shoulder_state)
    begin_d = int(len_data/100*1)
    end_d = int(len_data/100*25)

    fig = plt.figure()
    ax = fig.add_subplot()
    cb = ax.imshow(strainmap_1.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=0, vmax=1.7)
    ax.plot(np.rad2deg(estimated_shoulder_state[begin_d:end_d, 0]), np.rad2deg(estimated_shoulder_state[begin_d:end_d, 2]), linewidth=2)
    ax.scatter(np.array([45]), np.array([95]), label = 'start', c = 'red', s = 30)
    ax.scatter(np.array([60]), np.array([60]), label = 'goal', c='green', s = 30)
    ax.set_ylim(45, 110)
    ax.set_xlim(30, 63)
    ax.set_xlabel("Plane of Elevation [°]")
    ax.set_ylabel("Shoulder Elevation [°]")
    ax.legend()
    ax.set_title("EXP2: Trajectory on strainmap (initial condition)")
    fig.colorbar(cb)


    # visualize 2D trajectory on strainmap (1st landscape change)
    len_data = len(estimated_shoulder_state)
    begin_d = int(len_data/100*1)
    end_d = int(len_data/100*42)
    change_d = int(len_data/100*22)

    timestamp_change = estimated_shoulder_state[change_d, -1]
    old_plan = optimal_trajectory[:, np.abs(optimal_trajectory[0,:,-1] - timestamp_change).argmin(), :]

    fig = plt.figure()
    ax = fig.add_subplot()
    cb = ax.imshow(strainmap_2.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=0, vmax=1.7)
    ax.scatter(np.rad2deg(old_plan[0,2:-2]), np.rad2deg(old_plan[2,2:-2]), color = 'white', s = 2)
    ax.plot(np.rad2deg(estimated_shoulder_state[begin_d:end_d, 0]), np.rad2deg(estimated_shoulder_state[begin_d:end_d, 2]))
    ax.scatter(np.array([45]), np.array([95]), label = 'start', c = 'red', s = 30)
    ax.scatter(np.array([60]), np.array([60]), label = 'goal', c='green', s = 30)
    ax.set_ylim(58, 97)
    ax.set_xlim(43, 63)
    ax.set_xlabel("Plane of Elevation [°]")
    ax.set_ylabel("Shoulder Elevation [°]")
    ax.legend()
    ax.set_title("EXP2: Trajectory on updated strainmap")
    fig.colorbar(cb)

    # visualize 2D trajectory on strainmap (2nd landscape change)
    len_data = len(estimated_shoulder_state)
    begin_d = int(len_data/100*1)
    end_d = int(len_data/100*99.5)
    change_d = int(len_data/100*40)

    timestamp_change = estimated_shoulder_state[change_d, -1]
    old_plan = optimal_trajectory[:, np.abs(optimal_trajectory[0,:,-1] - timestamp_change).argmin(), :]

    fig = plt.figure()
    ax = fig.add_subplot()
    cb = ax.imshow(strainmap_3.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=0, vmax=1.7)
    ax.plot(np.rad2deg(estimated_shoulder_state[begin_d:end_d, 0]), np.rad2deg(estimated_shoulder_state[begin_d:end_d, 2]))
    ax.scatter(np.rad2deg(old_plan[0,2:-2]), np.rad2deg(old_plan[2,2:-2]), color = 'white', s = 2)
    ax.scatter(np.array([45]), np.array([95]), label = 'start', c = 'red', s = 30)
    ax.scatter(np.array([60]), np.array([60]), label = 'goal', c='green', s = 30)
    ax.set_ylim(58, 97)
    ax.set_xlim(43, 63)
    ax.set_xlabel("Plane of Elevation [°]")
    ax.set_ylabel("Shoulder Elevation [°]")
    ax.legend()
    ax.set_title("EXP2: Trajectory on updated strainmap")
    fig.colorbar(cb)

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
    ax.plot(xyz_curr[:,-1], xyz_curr[:,0], label = 'x_curr')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,0], label = 'x_cmd')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig.add_subplot(312)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,1], label = 'y_curr')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,1], label = 'y_cmd')
    ax.set_ylabel('[m]')
    ax.legend()
    ax = fig.add_subplot(313)
    ax.plot(xyz_curr[:,-1], xyz_curr[:,2], label = 'z_curr')
    ax.plot(xyz_cmd[:,-1], xyz_cmd[:,2], label = 'z_cmd')
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
    
    plt.show(block=True)


if __name__ == '__main__':
    main()