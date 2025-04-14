"""
Script to analyze the data collected in a rosbag during the tuning of cost function weights for the
"Passive human subject" case (Fig. 4 in the paper)

The bag files used in this script has not been included in the repository. To create data to visualize, 
run the TO_simulation.py and robot_control.py (in simulation mode) and set the parameters to test by editing
experiment_parameters.py. This is referred to as experiment 3 in experiment_parameters.py
"""

import os
import pickle
from spatialmath import SO3
import numpy as np
import rosbag
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
    path_to_bag = os.path.join(path_to_repo, 'Personal_Results', 'bags', 'experiment_4')

    bag_file_name = 'exp4_1.bag'

    # load the strainmap dictionary used in the experiment
    file_strainmaps = os.path.join(path_to_repo, 'Musculoskeletal Models', 'Strain Maps', 'Passive', 'differentiable_strainmaps_allTendons.pkl')

    # assume that the A* output has been interpolated to 50 datapoints
    n_interp = 50

    with rosbag.Bag(os.path.join(path_to_bag, bag_file_name), 'r') as bag:
        print('Extracting optimization outputs')
        optimal_trajectory = None
        for _, msg, time_msg in bag.read_messages(topics=['/optimization_output']):
            data = np.array(msg.data).reshape((6,50))
            time_curr = time_msg.to_time()
            if optimal_trajectory is None:
                optimal_trajectory = np.hstack((data, time_curr * np.ones((6,1))))
            else:
                optimal_trajectory = np.vstack((optimal_trajectory, np.hstack((data, time_curr * np.ones((6,1))))))

    strainmap = generate_approximated_strainmap(file_strainmaps, np.deg2rad(-2))

    # visualize 2D trajectory on strainmap
    fig = plt.figure()
    ax = fig.add_subplot()
    cb =ax.imshow(strainmap.T, origin='lower', cmap='hot', extent=[-20, 160, 0, 144], vmin=1.5, vmax=3.5)
    fig.colorbar(cb, ax=ax, label = 'Strain [%]')
    ax.scatter(np.array([120]), np.array([100]), label = 'start')
    ax.scatter(np.array([50]), np.array([40]), label = 'goal')
    # plot also the optimal trajectories
    ax.plot(np.rad2deg(optimal_trajectory[0,:-1]), np.rad2deg(optimal_trajectory[2,:-1]), color='black')
    ax.set_xlabel("Plane of elevation [deg]")
    ax.set_ylabel("Shoulder elevation [deg]")
    ax.legend()
    ax.axis('equal')
    ax.set_xlim(30, 130)
    ax.set_ylim(30, 120)
    ax.set_title("A* result")

    plt.show(block=True)

if __name__ == '__main__':
    main()