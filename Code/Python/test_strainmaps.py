# This script is dedicated to some testing related to the strainmaps in particular

import utilities_TO as utils_TO
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import casadi as ca

# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository

# retrieve the original (discrete) strainmaps
strainmaps_path = '/home/itbellix/Desktop/PTbot data/strainmaps'
file_name = 'IS_0_475.npy'                      # file containing strains as a function of time
file = os.path.join(strainmaps_path, file_name)
strainmaps = np.load(file)                      # 3D strainmaps: [time][plane_elev][shoulder_elev]
strainmap = strainmaps[20, :, :]                # consider the strain at a given time

# retrieve the parametrized (continuous) strainmaps
file_strainmaps_params = '/home/itbellix/Desktop/GitHub/PTbot_official/params_strainmaps.pkl'

with open(file_strainmaps_params, 'rb') as file:
    strainmaps_dict = pickle.load(file)

# set the known parameters for the maps (num of data-point, max and min,...)
max_se = 144; min_se = 0
max_pe = 160; min_pe = -20

pe_len, se_len = np.shape(strainmap)
step = 4

pe_datapoints = np.array(np.arange(min_pe, max_pe, step))
se_datapoints = np.array(np.arange(min_se, max_se, step))

X,Y = np.meshgrid(pe_datapoints, se_datapoints, indexing='ij')

# instantiate the NLP module to operate on the strains
my_nlps = utils_TO.nlps_module()
x = ca.MX.sym('x', 6)   # state vector: [theta, theta_dot, psi, psi_dot, phi, phi_dot], in rad or rad/s
my_nlps.initializeStateVariables(x = x, names = ['theta', 'theta_dot', 'psi', 'psi_dot', 'phi', 'phi_dot'])

# visualize the parametrized strainmap
my_nlps.setParametersStrainMap(strainmaps_dict['num_gaussians'], strainmaps_dict['all_params_gaussians'])
my_nlps.visualizeCurrentStrainMap()

# visualize the original discrete strainmap
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf1 = ax.plot_surface(X, Y, strainmap, cmap='plasma')
ax.set_zlim([0, strainmap.max()])
ax.set_xlabel('Plane Elev [deg]')
ax.set_ylabel('Shoulder Elev [deg]')
ax.set_zlabel('Strain level [%]')
plt.show(block=False)

# now, use the symbolic version of the strainmaps that are stored in the NLPS module
# parameters of the 1st Gaussian
amplitude_g1 = my_nlps.all_params_gaussians[0]
x0_g1 = my_nlps.all_params_gaussians[1]
y0_g1 = my_nlps.all_params_gaussians[2]
sigma_x_g1 = my_nlps.all_params_gaussians[3]
sigma_y_g1 = my_nlps.all_params_gaussians[4]
offset_g1 = my_nlps.all_params_gaussians[5]

# definition of the 1st Gaussian (note that the state variables are normalized!)
g1 = amplitude_g1 * np.exp(-((my_nlps.x[0]/np.max(np.abs(my_nlps.pe_boundaries))-x0_g1)**2/(2*sigma_x_g1**2) + (my_nlps.x[2]/np.max(np.abs(my_nlps.se_boundaries))-y0_g1)**2/(2*sigma_y_g1**2))) + offset_g1

# parameters of the 2nd Gaussian 
amplitude_g2 = my_nlps.all_params_gaussians[6]
x0_g2 = my_nlps.all_params_gaussians[7]
y0_g2 = my_nlps.all_params_gaussians[8]
sigma_x_g2 = my_nlps.all_params_gaussians[9]
sigma_y_g2 = my_nlps.all_params_gaussians[10]
offset_g2 = my_nlps.all_params_gaussians[11]

# definition of the 2nd Gaussian (note that the state variables are normalized!)
g2 = amplitude_g2 * np.exp(-((my_nlps.x[0]/np.max(np.abs(my_nlps.pe_boundaries))-x0_g2)**2/(2*sigma_x_g2**2) + (my_nlps.x[2]/np.max(np.abs(my_nlps.se_boundaries))-y0_g2)**2/(2*sigma_y_g2**2))) + offset_g2

# parameters of the 3rd Gaussian 
amplitude_g3 = my_nlps.all_params_gaussians[12]
x0_g3 = my_nlps.all_params_gaussians[13]
y0_g3 = my_nlps.all_params_gaussians[14]
sigma_x_g3 = my_nlps.all_params_gaussians[15]
sigma_y_g3 = my_nlps.all_params_gaussians[16]
offset_g3 = my_nlps.all_params_gaussians[17]

# definition of the 3rd Gaussian (note that the state variables are normalized!)
g3 = amplitude_g3 * np.exp(-((my_nlps.x[0]/np.max(np.abs(my_nlps.pe_boundaries))-x0_g3)**2/(2*sigma_x_g3**2) + (my_nlps.x[2]/np.max(np.abs(my_nlps.se_boundaries))-y0_g3)**2/(2*sigma_y_g3**2))) + offset_g3

# definition of the symbolic cumulative strainmap
strainmap_expression = g1 + g2 + g3
strainmap_sym = ca.Function('strainmap_sym', [my_nlps.x], [strainmap_expression])

# now, let's build the discrete strainmap again using this symbolic formulation
strainmap_recreated = np.zeros(my_nlps.X_norm.shape)
for pe_index in range(pe_datapoints.shape[0]):
    pe_value = pe_datapoints[pe_index]
    for se_index in range(se_datapoints.shape[0]):
        se_value = se_datapoints[se_index]
        data_point = ca.DM([pe_value, 0, se_value, 0, 0, 0])
        strainmap_recreated[pe_index, se_index] = strainmap_sym(data_point)

# visualize the original discrete strainmap
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf1 = ax.plot_surface(X, Y, strainmap_recreated, cmap='plasma')
ax.set_zlim([0, strainmap.max()])
ax.set_xlabel('Plane Elev [deg]')
ax.set_ylabel('Shoulder Elev [deg]')
ax.set_zlabel('Strain level [%]')
ax.set_title('Recreated strainmap')
plt.show()