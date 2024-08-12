# Testing script for the model produced in test_compute_inertias.py

import opensim as osim
import numpy as np
import os
import utilities_from_Simbody_copy as utils

## PARAMS -------------------------------------------------------------------------------------------------------
model_complete_name = 'debug_2bodies.osim'      # name of the full model that has all the required bodies
model_simplified_name = 'debug_1body.osim'      # name of the model with only one moving body
# ----------------------------------------------------------------------------------------------------------------
# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository
path_to_model = os.path.join(path_to_repo, 'OsimModels')    # getting path to the OpenSim models

# load OpenSim models (the original one, and the one updated with the results from test_compute_inertias.py)
model_complete = osim.Model(os.path.join(path_to_model, model_complete_name))
model_simplified = osim.Model(os.path.join(path_to_model, model_simplified_name))

# initialize simplified model and get its state
state_simpl = model_simplified.initSystem()

# add reserve actuators to both the complete (reference) and the new (test) models
# after having added them, set all the actuators to be overwritten
optimal_force = 1

# reference model
modelProcessor_ref = osim.ModelProcessor(os.path.join(path_to_model, model_complete_name))
modelProcessor_ref.append(osim.ModOpAddReserves(optimal_force))
model_ref = modelProcessor_ref.process()
state_ref = model_ref.initSystem()

acts_ref = model_ref.getActuators()
acts_ref_downcasted = []
for index_act in range(acts_ref.getSize()):
        act = osim.ScalarActuator.safeDownCast(acts_ref.get(index_act))
        act.overrideActuation(state_ref, True)
        acts_ref_downcasted.append(act)

# new model (to be tested)
modelProcessor_new = osim.ModelProcessor(os.path.join(path_to_model, model_simplified_name))
modelProcessor_new.append(osim.ModOpAddReserves(optimal_force))
model_new = modelProcessor_new.process()
state_new = model_new.initSystem()

acts_new = model_new.getActuators()
acts_new_downcasted = []
for index_act in range(acts_new.getSize()):
        act = osim.ScalarActuator.safeDownCast(acts_new.get(index_act))
        act.overrideActuation(state_new, True)
        acts_new_downcasted.append(act)

# get the number and indexes of the free coordinates in both models
num_free_coord_ref = 0
index_free_coords_ref = []
num_free_coord_new = 0
index_free_coords_new = []

for index_coord in range(model_ref.getNumCoordinates()):
     coord = model_ref.getCoordinateSet().get(index_coord)
     if coord.get_locked() is not True:
          num_free_coord_ref += 1
          index_free_coords_ref.append(index_coord)


for index_coord in range(model_new.getNumCoordinates()):
     coord = model_new.getCoordinateSet().get(index_coord)
     if coord.get_locked() is not True:
          num_free_coord_new += 1
          index_free_coords_new.append(index_coord)

assert num_free_coord_new ==  num_free_coord_ref, "The two models must have the same free coordinates!"

# create dummy generalized forces to be applied to both the models
nTests = 10
magnitudeGenForces = 10     # max magnitude of forces that will be applied
genForcesDummy = magnitudeGenForces*np.random.random((model_ref.getForceSet().getSize(), nTests))

# create dummy joint angles and velocities to set model state (they will be set to both models)
magnitudePositionPerturbation = 1
magnitudeSpeedPerturbation = 1
positionsDummy = magnitudePositionPerturbation*np.random.random((num_free_coord_ref, nTests))
speedsDummy = magnitudeSpeedPerturbation*np.random.random((num_free_coord_ref, nTests))

# set the model in a random configuration, apply generalized forces, solve for accelerations and store them
accelerations_ref = np.zeros((num_free_coord_ref, nTests))
accelerations_new = np.zeros((num_free_coord_new, nTests))

for test in range(nTests):
    # set joint angles and speed (for both models)
    for curr_index in range(num_free_coord_ref):
        # reference model
        model_ref.getCoordinateSet().get(index_free_coords_ref[curr_index]).setValue(state_ref, positionsDummy[curr_index, test])
        model_ref.getCoordinateSet().get(index_free_coords_ref[curr_index]).setSpeedValue(state_ref, speedsDummy[curr_index, test])
        
        # new model
        model_new.getCoordinateSet().get(index_free_coords_new[curr_index]).setValue(state_new, positionsDummy[curr_index, test])
        model_new.getCoordinateSet().get(index_free_coords_new[curr_index]).setSpeedValue(state_new, speedsDummy[curr_index, test])

    # apply the same generalized forces to reference and new model
    for index_act in range(acts_ref.getSize()):
        acts_ref_downcasted[index_act].setOverrideActuation(state_ref, genForcesDummy[index_act, test])
        acts_new_downcasted[index_act].setOverrideActuation(state_new, genForcesDummy[index_act, test])

    # realize the model to acceleration stage (solve forward dynamics)
    model_ref.realizeAcceleration(state_ref)
    model_new.realizeAcceleration(state_new)

    # store the accelerations for both models, induced in the same configurations by the same forces
    for curr_index in range(num_free_coord_ref):
        accelerations_ref[curr_index, test] = model_ref.getCoordinateSet().get(index_free_coords_ref[curr_index]).getAccelerationValue(state_ref)
        accelerations_new[curr_index, test] = model_new.getCoordinateSet().get(index_free_coords_new[curr_index]).getAccelerationValue(state_new)

# Verify accelerations from reference and new model (they must be the same!)
diff = np.abs(accelerations_ref - accelerations_new)
if np.min(diff) > 1e-10:         # leave some tolerance if some rounding happens
    raise ValueError("Acceleration verification test failed -> the new model is wrong...")    
print('Acceleration verification test passed -> the new model is correct!')
