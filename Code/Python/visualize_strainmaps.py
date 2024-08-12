# This script allows to visualize the strainmaps for a given muscle

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle

code_path = os.path.dirname(os.path.realpath(__file__))
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository

## PARAMETERS------------------------------------------------------

# define the required paths (relative to path_to_repo)
strainmaps_path = 'Personal_Results/Strains/Passive/AllMuscles'

# file containing all the strains (when muscles are relaxed)
file_name = 'All_0.npy'

# do we want to save the 2D figures of the strainmaps?
save_2D = True

# do we want to visualize the 3D strainmaps?
viz_3D = True

# create a folder where to save the results (and the plots, if print_flag is True)
folder_img, _ = os.path.splitext(file_name)
os.makedirs(os.path.join(path_to_repo, strainmaps_path, folder_img), exist_ok=True)

## ----------------------------------------------------------------

# load strainmap to visualize
# Acronyms used in the following are:
# * PE: plane of elevation
# * SE: shoulder elevation
# * AR: axial rotation

# boundary values for SE [deg] (coming from OpenSim model, to include only feasible poses)
max_se = 144
min_se = 0

# boundary values for PE [deg] (coming from OpenSim model, to include only feasible poses)
max_pe = 160
min_pe = -20

# boundary values for AR [deg] (coming from OpenSim model, to include only feasible poses)
min_ar = -90
max_ar = 102

# step considered across all coordinates when retrieving the strain
step = 4    # This means that the model's strain was retrieved from the model combining the values
            # of the glenohumeral coordinates in increments of "step" degrees

# dimension of a single strain map (where axial rotation is fixed)
pe_len = np.shape(np.arange(min_pe, max_pe, step))[0]
se_len = np.shape(np.arange(min_se, max_se, step))[0]
ar_len = np.shape(np.arange(min_ar, max_ar, step))[0]       # compute also the length across this dimension

# create the grids that will be used for interpolation (and visualization, if needed)
pe_datapoints = np.array(np.arange(min_pe, max_pe, step))
se_datapoints = np.array(np.arange(min_se, max_se, step))

X,Y = np.meshgrid(pe_datapoints, se_datapoints, indexing='ij')

# load the strainmaps from file (stores strains as functions of se, pe, ar coordinates)
file = os.path.join(path_to_repo, strainmaps_path, file_name)
strainmaps = np.load(file)                      # 3D strainmaps: [AR][PE][SE]

# considering one value of axial rotation at a time, visualize all the strainmaps
for ar_index in range(ar_len):
    # select the strainmap corresponding to the current index
    strainmap = strainmaps[ar_index, :, :]

    figure_title = 'Axial rotation: {}'.format(min_ar + step * ar_index)
    figure_name = 'Axial_rot_{}.pdf'.format(min_ar + step * ar_index)

    # plot strainmap in 2D and save the image to the new folder
    if save_2D:
        fig = plt.figure(ar_index)
        ax = fig.add_subplot()
        heatmap = ax.imshow(np.flip(strainmap.T, axis = 0), cmap = 'hot', extent=[min_pe, max_pe, min_se, max_se])
        fig.colorbar(heatmap, label='Strain level [%]')
        ax.set_xlabel('Plane Elev [deg]')
        ax.set_xticks(np.arange(min_pe, max_pe + 1, 20))
        ax.set_xticklabels(np.arange(min_pe, max_pe + 1, 20).astype(int))
        ax.set_ylabel('Shoulder Elev [deg]')
        ax.set_yticks(np.arange(min_se, max_se + 1, 20))
        ax.set_yticklabels(np.arange(min_se, max_se + 1, 20).astype(int))
        ax.set_title(figure_title)
        fig.tight_layout()
        fig.savefig(os.path.join(path_to_repo, strainmaps_path, folder_img, figure_name))
        plt.close(fig)

    # plot strainmap in 3D and visualize it for inspection
    if viz_3D:
        fig = plt.figure(ar_index)
        ax = fig.add_subplot(projection='3d')
        surf1 = ax.plot_surface(X, Y, strainmap, cmap='plasma')
        ax.set_xlabel('Plane Elev [deg]')
        ax.set_ylabel('Shoulder Elev [deg]')
        ax.set_zlabel('Strain level [%]')
        ax.set_zlim([0, strainmap.max()])
        ax.set_title(figure_title)

plt.show()
