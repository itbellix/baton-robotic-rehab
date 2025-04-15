# This script is used to extract the strain information from the active strainmaps that Irene generated in her
# Master's thesis. Originally, a different file was generated for each muscle, at a fixed muscle activation level.
# This script processes the original files, and combines them in a 4D strainmap for each muscle.
# The user needs to indicate the folder where the 3D strainmaps are saved, and the the acronym for the muscle 
# to analyze.

import os
import numpy as np

code_path = os.path.dirname(os.path.realpath(__file__))
path_to_repo = os.path.join(code_path, '..')          # getting path to the repository

## PARAMETERS------------------------------------------------------

# define the required paths
strainmaps_path = '/home/itbellix/Desktop/PTbot/Strainmaps/Generic2019_noload_all_sections'

# muscle to analyze
muscle_of_interest = 'SSCS'

# list of codenames for the muscles (we have information only on the rotator cuff)
# ISI = infraspinatus inferior
# ISS = infraspinatus superior
# SSCI = subscapularis inferior
# SSCM = subscapularis medialis
# SSCS = subscapularis superior
# SSPA = supraspinatus anterior
# SSPP = supraspinatus posterior
# TM = teres minor

# -------------------------------------------------------------------------------
# select only the files containing info on the muscle we want
file_list = []

for file in os.listdir(strainmaps_path):
    if file.startswith(muscle_of_interest) and file.endswith('.npy'):
        file_list.append(file)

# Sort the filenames based on the last part (the number, which indicates the muscle activation)
file_list.sort(key=lambda x: float(x.split('_')[-1].split('.')[0]) /
                                (10 if len(x.split('_')[-1].split('.')[0]) == 1 else
                                 (100 if len(x.split('_')[-1].split('.')[0]) == 2 else 1000)))

# The strainmaps will capture the strain at given positions in the shoulder state,
# defined by the following coordinates:
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

num_files = len(file_list)
strainmap_4d = np.zeros((ar_len, pe_len, se_len, num_files))

for index_file in range(num_files):
    strainmap_3d_current = np.load(os.path.join(strainmaps_path, file_list[index_file]))
    print('appending ', file_list[index_file])
    strainmap_4d[:, :, :, index_file] = strainmap_3d_current

# save the resulting 4D maps
file_name = muscle_of_interest + '_active.npy'
np.save(os.path.join(strainmaps_path, file_name), strainmap_4d)