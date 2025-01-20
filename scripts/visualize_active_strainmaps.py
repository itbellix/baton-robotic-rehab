"""
This script allow to visualize the 4D strain maps along a different direction.
The 4D strain map carries information of the strain of a specific tendon of the rotator cuff
as a function of the axial rotation, plane of elevation, and shoulder elevation of the glenohumeral joint, 
as well as muscle activation.
For ease of visualization, we present the strain maps on a 2D plane (plane of elevation vs shoulder elevation).
Here, we provide a tool to quickly span along the other two directions (fixing one of the two),
and visually explore how the strain changes there.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle

code_path = os.path.dirname(os.path.realpath(__file__))
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository

## PARAMETERS------------------------------------------------------

# define the required paths (relative to path_to_repo)
strainmaps_path = 'Personal_Results/Strains/Active'

# muscle to analyze
muscle_of_interest = 'SSPA'     # 'SSPA': supraspinatus anterior
                                # 'ISI': infraspinatus inferior

# list of codenames for the muscles (we have information only on the rotator cuff)
# ISI = infraspinatus inferior
# ISS = infraspinatus superior
# SSCI = subscapularis inferior
# SSCM = subscapularis medialis
# SSCS = subscapularis superior
# SSPA = supraspinatus anterior
# SSPP = supraspinatus posterior
# TM = teres minor

# visualization_mode
viz_mode = 1    # 1: fix axial rotation and span activation
                # 2: fix activation and span axial rotation
# -------------------------------------------------------------------------------
file_name = muscle_of_interest + '_active.npy'

strainmap_4d = np.load(os.path.join(path_to_repo, strainmaps_path, file_name))

# The 4D strain map carries information of the strain as a function of the axial rotation, plane of elevation, and
# shoulder elevation of the glenohumeral joint, as well as muscle activation.
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

# boundary value for the activation (it is a variable that should belong to the 0-1 interval)
min_activ = 0
max_activ = 0.5

# step considered across all coordinates when retrieving the strain
step_coord = 4  # This means that the model's strain was retrieved from the model combining the values
                # of the glenohumeral coordinates in increments of "step" degrees

step_activ = 0.005  # this means that the activation increases by this amount every time

# dimension of a single strain map (where axial rotation is fixed)
pe_len = np.shape(np.arange(min_pe, max_pe, step_coord))[0]
se_len = np.shape(np.arange(min_se, max_se, step_coord))[0]
ar_len = np.shape(np.arange(min_ar, max_ar, step_coord))[0]       # compute also the length across this dimension

# VISUALIZE the variation of tendon strain in 2D maps
# Only plane of elevation and shoulder elevation will be shown
# Values of axial rotation are increased gradually, once they are set the whole variation caused by muscle activation is shown
frequency = 60 # update frequency in Hz, to update the portion of strain map visualized

# find the maximum strain for the tendon, to scale the color in the map
max_strain = np.max(np.max(strainmap_4d))

# Create the figure and axis for the plot
fig, ax = plt.subplots()
heatmap = ax.imshow(strainmap_4d[0, :, :, 0].T, origin='lower', extent=[min_pe, max_pe, min_se, max_se], vmin=0, vmax=max_strain, cmap = 'hot')

cb = fig.colorbar(heatmap, label='Strain level [%]')

ax.set_xlabel('Plane Elev [deg]')
ax.set_xticks(np.arange(min_pe, max_pe + 1, 20))
ax.set_xticklabels(np.arange(min_pe, max_pe + 1, 20).astype(int))
ax.set_ylabel('Shoulder Elev [deg]')
ax.set_yticks(np.arange(min_se, max_se + 1, 20))
ax.set_yticklabels(np.arange(min_se, max_se + 1, 20).astype(int))

plt.pause(2)

if viz_mode == 1:
    # fix axial rotation and span activation
    for i in range(15, strainmap_4d.shape[0]-15):  # Loop through the first dimension (axial rotation), around 0 degrees
        for j in range(0, strainmap_4d.shape[3], 4):  # Loop through the fourth dimension ()
            title = '(' +muscle_of_interest+ ') AR = ' + str(min_ar + step_coord * i) + '  activation = ' +  format(np.round(min_activ + step_activ * j, 3), '.3f')
            heatmap.set_array(strainmap_4d[i, :, :, j].T)  # Update the data
            plt.pause(1 / frequency)  # Pause to control the update frequency
            ax.set_title(title)
            plt.draw()
elif viz_mode == 2:
    # fix activation and span axial rotation
    for j in range(0, strainmap_4d.shape[3], 20):
        for i in range(0, strainmap_4d.shape[0]):
            title = '(' +muscle_of_interest+ ') AR = ' + str(min_ar + step_coord * i) + '  activation = ' +  format(np.round(min_activ + step_activ * j, 3), '.3f')
            heatmap.set_array(strainmap_4d[i, :, :, j].T)  # Update the data
            plt.pause(1 / frequency)  # Pause to control the update frequency
            ax.set_title(title)
            plt.draw()
