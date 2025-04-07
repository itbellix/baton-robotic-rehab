"""
This script interpolates a strain map, through Radial Basis Function interpolation.
The single strain-map is interpolated with a sum of 2-dimensional Gaussians G_i(x,y), defined as:
G_i(x,y) = A * np.exp( -((x-x0)/xalpha)**2 -((y-y0)/yalpha)**2)

The user can indicate the maximum fitting error required (according to which the number of Gaussians
is determined), or the fixed number of Gaussians to be used in the interpolation of each 2D-layer.

The parameters of these Gaussians are identified, printed to screen, and optionally saved 
(to "params_strainmaps.pkl").

Note: the parameters of the Gaussians are found approximating a "normalized strain map".
      This facilitates the fitting, but requires to remember that the x-y coordinates indicating
      plane of elevation and shoulder elevation have been normalized dividing them by their 
      respective maximum value.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pickle

code_path = os.path.dirname(os.path.realpath(__file__))
path_to_repo = os.path.join(code_path, '..')          # getting path to the repository

## PARAMETERS -----------------------------------------------------------------------------------------------
# choose if we want to fit a given number of Gaussians, or we want to control the maximum fitting RMSE.
# In the paper, we opted for fitting 3 Gaussians per strain map, since the RMSE was very low on the strain maps that
# we operated on. Also, this allow for some advantages in the downstream implementation.

max_fitting_error_fixed = None      # maximum allowable RMS error between original and fitted map
num_gaussians_fixed = 3             # number of Gaussians to be used in the interpolation
save_prms_to_file =  True           # should the resulting parameters be saved to a file? (True/False)
print_flag = True                   # should the results of the fitting process be visualized? (True/False)
file_prms= 'params_strainmaps'      # the file to which the parameters will be saved (change the name if needed)

fit_only_central_ar = True          # fitting only around axial rotation = 0, not to waste time

# define the required paths (relative to path_to_repo)
strainmaps_path = 'Musculoskeletal Models/Strain Maps/Passive'

# file containing all the strains (when muscles are relaxed)
file_name = 'All_0.npy'       # This file is provided in the repo as an example. If other files are necessary, they
                                    # should be constructed from the OpenSim musculoskeletal model directly (available at https://simtk.org/projects/thoracoscapular)

#------------------------------------------------------------------------------------------------------------

# check that the initialization of the parameters make sense
assert max_fitting_error_fixed is None or num_gaussians_fixed is None, "Choose if we want to control the maximum error \
                                                                  or the number of fitting functions"

# define the name of the resulting file, as a function of the parameters set
if max_fitting_error_fixed is not None:
    decimal_places = len(str(max_fitting_error_fixed).split('.')[1])
    numeric_part = f'{max_fitting_error_fixed:.{decimal_places}f}'.replace('.', '')
    specific_name = f'_max_RMSE_{numeric_part}'
else:
    specific_name = f'_num_Gauss_{num_gaussians_fixed}'

folder_prms = file_prms + specific_name
file_prms = file_prms + specific_name + '.pkl'

# create a folder where to save the results (and the plots, if print_flag is True)
os.makedirs(os.path.join(path_to_repo, strainmaps_path, folder_prms), exist_ok=False)

## Define the functions used for the 2D interpolation -------------------------------------------------------------------
num_params = 6      # number of params in the Gaussian_2d function (amplitude, x0, y0, sigma_x, sigma_y, offset)
                    # we do not consider explicitly of the rotation of the principal axis.

def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y, offset):
    return amplitude * np.exp(-((x-x0)**2/(2*sigma_x**2)+(y-y0)**2/(2*sigma_y**2)))+offset


# This is the callable function that is passed to the curve_fit routine
def _gaussian_2d(grid, *args):
    x, y = grid
    arr = np.zeros(x.shape)
    for i in range(len(args)//num_params):
       arr += gaussian_2d(x, y, *args[i*num_params:i*num_params+num_params])

    # add this term to penalize unrealistic fitting
    if arr.min() < 0:
        constraint_violation = 2 * arr.min()**2
    else:
        constraint_violation = 0

    return arr + constraint_violation
# ----------------------------------------------------------------------------------------------------------------------

# load strainmap to operate with
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

# boundary values for MUSCLE_ACTIVATION
min_musc_act = 0
max_musc_act = 0.5
step_musc_act = 0.005

# dimension of a single strain map (where axial rotation is fixed)
pe_len = np.shape(np.arange(min_pe, max_pe, step))[0]
se_len = np.shape(np.arange(min_se, max_se, step))[0]
musc_act_len = np.shape(np.arange(min_musc_act, max_musc_act, step_musc_act))[0]
ar_len = np.shape(np.arange(min_ar, max_ar, step))[0]       # compute also the length across this dimension

# create the grids that will be used for interpolation (and visualization, if needed)
pe_datapoints = np.array(np.arange(min_pe, max_pe, step))
se_datapoints = np.array(np.arange(min_se, max_se, step))

X,Y = np.meshgrid(pe_datapoints, se_datapoints, indexing='ij')
X_norm = X/max_pe
Y_norm = Y/max_se
data = np.vstack((X.ravel(), Y.ravel()))
data_norm = np.vstack((X_norm.ravel(), Y_norm.ravel()))

# load the strainmaps from file
file = os.path.join(path_to_repo, strainmaps_path, file_name)
strainmaps = np.load(file)

# check strainmap dimensions
if len(strainmaps.shape) == 3:
    threeD = True               # 3D strainmaps: [AR][PE][SE]
    fourD = False
if len(strainmaps.shape) == 4:
    threeD = False
    fourD = True                # 4D strainmaps: [AR][PE][SE][MUSCLE_ACTIVATION]

# select which portion of the axial rotation span we are interested in
if fit_only_central_ar:
    ar_index_delta = 17
else:
    ar_index_delta = 0         

if threeD:
    # allocate the lists to store the results
    num_gaussians = np.zeros((ar_len, 1))               # number of Gaussians used to interpolate the i-th map
    all_params_gaussians = [[] for _ in range(ar_len)]  # parameters of all the Gaussians used for the i-th map
    error_per_map = np.zeros((ar_len, 1))               # RMSE error from the fitting, for every map

    # considering one value of axial rotation at a time, interpolate all of the strainmaps
    for ar_index in range(ar_index_delta, ar_len-ar_index_delta):
        print("\n\nIteration #", ar_index)

        # select the strainmap corresponding to the current index
        strainmap = strainmaps[ar_index, :, :]

        # initialize the parameters for running the fitting
        guess_single_gaussian = 0.5*np.ones(6)
        error = np.inf
        guess_prms = []

        # discriminate whether we want fixed number of Gaussians, or fixed maximum error
        if max_fitting_error_fixed is not None:
            while error > max_fitting_error_fixed:
                guess_prms.append(guess_single_gaussian)

                # Flatten the initial guess parameter list.
                p0 = [p for prms in guess_prms for p in prms]

                popt, pcov = opt.curve_fit(_gaussian_2d, data_norm, strainmap.ravel(), p0, bounds=np.vstack((-100*np.ones(np.shape(p0)), 100*np.ones(np.shape(p0)))) ,maxfev=10**5)

                fit = np.zeros(strainmap.shape)
                for i in range(len(popt)//num_params):
                    fit += gaussian_2d(X_norm, Y_norm, *popt[i*num_params:i*num_params+num_params])

                error = np.sqrt(np.mean((strainmap - fit) ** 2))   # rms error

        if num_gaussians_fixed is not None:
            for index_gaussian in range(num_gaussians_fixed):
                guess_prms.append(guess_single_gaussian)

                # Flatten the initial guess parameter list.
                p0 = [p for prms in guess_prms for p in prms]

                popt, pcov = opt.curve_fit(_gaussian_2d, data_norm, strainmap.ravel(), p0, bounds=np.vstack((-100*np.ones(np.shape(p0)), 100*np.ones(np.shape(p0)))) ,maxfev=10**5)

                fit = np.zeros(strainmap.shape)
                for i in range(len(popt)//num_params):
                    fit += gaussian_2d(X_norm, Y_norm, *popt[i*num_params:i*num_params+num_params])

                error = np.sqrt(np.mean((strainmap - fit) ** 2))   # rms error
        
        # save the optimal parameters for the fitting of the current strainmap
        num_gaussians[ar_index] = len(popt)//num_params
        all_params_gaussians[ar_index] = popt
        error_per_map[ar_index] = error

        if print_flag:
            print('Gaussians required:')
            print(len(popt)//num_params)
            print('RMSE:')
            print(error)

            figure_title = 'Axial rotation: {}'.format(min_ar + step * ar_index)
            figure_name = 'Axial_rot_{}.png'.format(min_ar + step * ar_index)

            # plot strainmap in 3D together with the fitted function
            fig = plt.figure(ar_index)
            ax = fig.add_subplot(projection='3d')
            surf1 = ax.plot_surface(X, Y, strainmap, cmap='plasma')
            surf2 = ax.plot_surface(X, Y, fit, label='fitted function', cmap='gray')
            surf2._edgecolors2d = surf2._edgecolor3d
            surf2._facecolors2d = surf2._facecolor3d
            cset = ax.contourf(X, Y, strainmap - fit, zdir='z', offset=0, cmap='plasma')
            ax.set_xlabel('Plane Elev [deg]')
            ax.set_ylabel('Shoulder Elev [deg]')
            ax.set_zlabel('Strain level [%]')
            ax.set_zlim([0, strainmap.max()])
            ax.set_title(figure_title)
            ax.legend()

            fig.savefig(os.path.join(path_to_repo, strainmaps_path, folder_prms, figure_name))
            plt.close(fig)

if fourD:
    num_gaussians = np.zeros((ar_len, musc_act_len))        # number of Gaussians used to interpolate the i-th map
    all_params_gaussians = [[] for _ in range(musc_act_len)]      # parameters of all the Gaussians used for the i-th map
    all_params_gaussians = [all_params_gaussians[:] for _ in range(ar_len)]   # consider axial rotation
    error_per_map = np.zeros((ar_len, musc_act_len))        # RMSE error from the fitting, for every map

    # initialize as None the results of the optimization
    popt = None

    # considering one value of axial rotation at a time, interpolate all of the strainmaps
    for ar_index in range(ar_index_delta, ar_len-ar_index_delta):
        print("\n\nIteration #", ar_index)

        for musc_act_index in range(musc_act_len):
            # select the strainmap corresponding to the current index
            strainmap = strainmaps[ar_index, :, :, musc_act_index]

            # initialize the parameters for running the fitting
            guess_single_gaussian = 0.5*np.ones(6)
            error = np.inf
            guess_prms = []

            # discriminate whether we want fixed number of Gaussians, or fixed maximum error
            if max_fitting_error_fixed is not None:
                while error > max_fitting_error_fixed:
                    guess_prms.append(guess_single_gaussian)

                    # Flatten the initial guess parameter list.
                    p0 = [p for prms in guess_prms for p in prms]

                    lb = -100*np.ones(np.shape(p0))
                    ub = 100*np.ones(np.shape(p0))

                    popt, pcov = opt.curve_fit(_gaussian_2d, data_norm, strainmap.ravel(), p0, bounds=np.vstack((lb, ub)) ,maxfev=10**5)

                    fit = np.zeros(strainmap.shape)
                    for i in range(len(popt)//num_params):
                        fit += gaussian_2d(X_norm, Y_norm, *popt[i*num_params:i*num_params+num_params])

                    error = np.sqrt(np.mean((strainmap - fit) ** 2))   # rms error


            if num_gaussians_fixed is not None:
                if popt is None:
                    # for the first iteration, use a general initial guess
                    guess_prms = np.tile(guess_single_gaussian, num_gaussians_fixed)
                elif ar_index>ar_index_delta:
                    # if we have already a solution for a previous axial rotation at the same activation, use
                    # it as initial guess
                    guess_prms = all_params_gaussians[ar_index-1][musc_act_index]
                else:
                    # otherwise, take as initial guess the result of the optimization at the previous level of muscle
                    # activation
                    guess_prms = popt

                    # Flatten the initial guess parameter list.
                p0 = guess_prms.tolist()

                lb = -100*np.ones(np.shape(p0))
                ub = 100*np.ones(np.shape(p0))

                popt, pcov = opt.curve_fit(_gaussian_2d, data_norm, strainmap.ravel(), p0, bounds=np.vstack((lb, ub)) ,maxfev=10**5)

                fit = np.zeros(strainmap.shape)
                for i in range(len(popt)//num_params):
                    fit += gaussian_2d(X_norm, Y_norm, *popt[i*num_params:i*num_params+num_params])

                error = np.sqrt(np.mean((strainmap - fit) ** 2))   # rms error
            
            # save the optimal parameters for the fitting of the current strainmap
            num_gaussians[ar_index, musc_act_index] = len(popt)//num_params
            all_params_gaussians[ar_index][musc_act_index] = popt
            error_per_map[ar_index, musc_act_index] = error

            if print_flag:
                print('    ', np.round(musc_act_index/musc_act_len,3) ,' Gaussians required: ', len(popt)//num_params)
                print('        RMSE: ', np.round(error,4))

                figure_title = 'Axial rotation:_' + str(min_ar + step * ar_index)
                figure_name = 'Axial_rot_' + str(min_ar + step * ar_index)+'_musc_act_'+ str(min_musc_act + step_musc_act * musc_act_index)+'.png'

                # plot strainmap in 3D together with the fitted function
                if musc_act_index%20==0:
                    fig = plt.figure(ar_index)
                    ax = fig.add_subplot(projection='3d')
                    surf1 = ax.plot_surface(X, Y, strainmap, cmap='plasma')
                    surf2 = ax.plot_surface(X, Y, fit, label='fitted function', cmap='gray')
                    surf2._edgecolors2d = surf2._edgecolor3d
                    surf2._facecolors2d = surf2._facecolor3d
                    cset = ax.contourf(X, Y, strainmap - fit, zdir='z', offset=0, cmap='plasma')
                    ax.set_xlabel('Plane Elev [deg]')
                    ax.set_ylabel('Shoulder Elev [deg]')
                    ax.set_zlabel('Strain level [%]')
                    ax.set_zlim([0, strainmap.max()])
                    ax.set_title(figure_title)
                    ax.legend()
                    fig.savefig(os.path.join(path_to_repo, strainmaps_path, folder_prms, figure_name))
                    plt.close(fig)


# evaluate the maximum number of interpolating Gaussians required, and the maximum RMSE
# over all the maps
max_num_gaussians = np.max(num_gaussians)
max_RMSE = np.max(error_per_map)
mean_RMSE = np.mean(error_per_map)

# print general info about the fitting
print("\nNumber of Gaussians overall")
print(num_gaussians)
print('Max number of Gaussians')
print(max_num_gaussians)
print('Max RMSE')
print(max_RMSE)
print('Mean RMSE')
print(mean_RMSE)

# save the parameters of the interpolation to file
if save_prms_to_file:
    dict = {}
    dict['num_gaussians'] = num_gaussians
    dict['all_params_gaussians'] = all_params_gaussians
    dict['max_RMSE_fit'] = max_fitting_error_fixed
    dict['num_gaussians_fixed'] = num_gaussians_fixed
    dict['error_per_map'] = error_per_map
    dict['max_RMSE'] = max_RMSE
    dict['mean_RMSE'] = mean_RMSE
    dict['active_strains'] = fourD

    with open(os.path.join(path_to_repo, strainmaps_path, folder_prms,file_prms), 'wb') as file:
        pickle.dump(dict, file)
