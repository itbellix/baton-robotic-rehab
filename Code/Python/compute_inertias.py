# In this script I want to get from a complete model the equivalent inertia of the arm (with elbow locked). 
# The inertial properties will be assigned to a simpler model, so that the correct dynamics of the human
# arm is captured by it (while making use of less Bodies in OpenSim). The new required model is generated
# and tested against the original one, to validate the inertial parameters that are identified.
#
# As a complete starting model I choose the thoracoscapular shoulder model (Seth 2019) 
# (where I lock all the coordinates apart from plane of elevation, shoulder elevation and scapula winging).

import opensim as osim
import numpy as np
import os
import utilities_from_Simbody as utils

## PARAMS -------------------------------------------------------------------------------------------------------
model_complete_name = 'right_arm_GH_full.osim'          # name of the full model that has all the required bodies
model_simplified_name = 'right_arm_GH_noForearm.osim'   # name of the model with only one moving body
model_new_name = 'right_arm_GH_noForearm_full.osim'     # name of the model that will be produced 
                                                        # (only one moving body, but with full inertial properties)

# ----------------------------------------------------------------------------------------------------------------
# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..', '..')             # getting path to the repository
path_to_model = os.path.join(path_to_repo, 'OsimModels')    # getting path to the OpenSim models

# load OpenSim models (the one to extract the parameters from , and the one to modify)
model_complete = osim.Model(os.path.join(path_to_model, model_complete_name))
model_simplified = osim.Model(os.path.join(path_to_model, model_simplified_name))

# from the complete model, retrieve the bodies composing the arm
body_humerus = model_complete.updBodySet().get('humerus')
body_ulna = model_complete.updBodySet().get('ulna')
body_radius = model_complete.updBodySet().get('radius')
body_hand = model_complete.updBodySet().get('hand')

# initialize the complete model to work on it, and get its state
state = model_complete.initSystem()

# ------- FINDING THE INERTIA OF THE HUMAN ARM in the required configuration -----------------------------------------------------------------------------
# Get the inertia of the arm (humerus+ulna+radius+hand) in the current configuration (elbow locked at 90 degree) 
# and express it in the humerus frame.
#   1. retrieve the inertias of the bodies, expressed in their reference frames
#   2. find the position of the center of mass of the human arm in the current configuration
#   3. "translate" the inertias to the the same point (i.e., the center of mass of the human arm)
#   4. express all the inertias in the same frame (fixed to the humerus), and sum them together
#
# In this process, we need to account for the fact that all of the inertia properties should be expressed with
# respect to the new body.
ground_frame = model_complete.get_ground()

# 1.  Note that inertias are expressed about the mass center, not the origin of the body frames!
#     Also, each inertia is expressed in a different (body-fixed) frame
humerus_inertia = body_humerus.get_inertia()
humerus_inertia_simbody = osim.simbody.Inertia(humerus_inertia.get(0), humerus_inertia.get(1), humerus_inertia.get(2), humerus_inertia.get(3), humerus_inertia.get(4), humerus_inertia.get(5))
humerus_mass = body_humerus.getMass()

ulna_inertia = body_ulna.get_inertia()
ulna_inertia_simbody = osim.simbody.Inertia(ulna_inertia.get(0), ulna_inertia.get(1), ulna_inertia.get(2), ulna_inertia.get(3), ulna_inertia.get(4), ulna_inertia.get(5))
ulna_mass = body_ulna.getMass()

radius_inertia = body_radius.get_inertia()
radius_inertia_simbody = osim.simbody.Inertia(radius_inertia.get(0), radius_inertia.get(1), radius_inertia.get(2), radius_inertia.get(3), radius_inertia.get(4), radius_inertia.get(5))
radius_mass = body_radius.getMass()

hand_inertia = body_hand.get_inertia()
hand_inertia_simbody = osim.simbody.Inertia(hand_inertia.get(0), hand_inertia.get(1), hand_inertia.get(2), hand_inertia.get(3), hand_inertia.get(4), hand_inertia.get(5))
hand_mass = body_hand.getMass()

arm_mass = humerus_mass + radius_mass + ulna_mass + hand_mass

# 2. Find the position of the CoM of the human arm
# find the locations of the body CoMs (in ground)
humerus_com_locationInGround = body_humerus.findStationLocationInGround(state, body_humerus.getMassCenter()).to_numpy()
ulna_com_locationInGround = body_ulna.findStationLocationInGround(state, body_ulna.getMassCenter()).to_numpy()
radius_com_locationInGround = body_radius.findStationLocationInGround(state, body_radius.getMassCenter()).to_numpy()
hand_com_locationInGround = body_hand.findStationLocationInGround(state, body_hand.getMassCenter()).to_numpy()

arm_com_locationInGround = (humerus_mass*humerus_com_locationInGround + ulna_mass*ulna_com_locationInGround +
                            radius_mass*radius_com_locationInGround + hand_mass*hand_com_locationInGround) / arm_mass

# find the location of the new center of mass, in the humerus reference frame
arm_com_locationInGround_vec3 = osim.Vec3(arm_com_locationInGround[0], arm_com_locationInGround[1], arm_com_locationInGround[2])
arm_com_locationInHum_vec3 = ground_frame.findStationLocationInAnotherFrame(state, arm_com_locationInGround_vec3, body_humerus)

# 3. Translate all of the inertias to the center of mass of the overall multibody system
distance_hum_comArm = arm_com_locationInGround - humerus_com_locationInGround
distance_hum_comArm_vec3 = osim.Vec3(distance_hum_comArm[0], distance_hum_comArm[1], distance_hum_comArm[2])

distance_ulna_comArm = arm_com_locationInGround - ulna_com_locationInGround
distance_ulna_comArm_vec3 = osim.Vec3(distance_ulna_comArm[0], distance_ulna_comArm[1], distance_ulna_comArm[2])

distance_radius_comArm = arm_com_locationInGround - radius_com_locationInGround
distance_radius_comArm_vec3 = osim.Vec3(distance_radius_comArm[0], distance_radius_comArm[1], distance_radius_comArm[2])

distance_hand_comArm = arm_com_locationInGround - hand_com_locationInGround
distance_hand_comArm_vec3 = osim.Vec3(distance_hand_comArm[0], distance_hand_comArm[1], distance_hand_comArm[2])

# express the distances in the correct body frames
translation_hum_comArm_inHumFrame = ground_frame.expressVectorInAnotherFrame(state, distance_hum_comArm_vec3, body_humerus).to_numpy()
translation_ulna_comArm_inUlnaFrame = ground_frame.expressVectorInAnotherFrame(state, distance_ulna_comArm_vec3, body_ulna).to_numpy()
translation_radius_comArm_inRadiusFrame = ground_frame.expressVectorInAnotherFrame(state, distance_radius_comArm_vec3, body_radius).to_numpy()
translation_hand_comArm_inRadiusFrame = ground_frame.expressVectorInAnotherFrame(state, distance_hand_comArm_vec3, body_hand).to_numpy()

# update inertias to be all expressed at the same point - center of mass of the arm - but still body frames
hum_inertia_translated_inHumFrame = utils.shiftFromMassCenter(humerus_inertia_simbody, translation_hum_comArm_inHumFrame, humerus_mass)
ulna_inertia_translated_inUlnaFrame = utils.shiftFromMassCenter(ulna_inertia_simbody, translation_ulna_comArm_inUlnaFrame, ulna_mass)
radius_inertia_translated_inRadiusFrame = utils.shiftFromMassCenter(radius_inertia_simbody, translation_radius_comArm_inRadiusFrame, radius_mass)
hand_inertia_translated_inHandFrame = utils.shiftFromMassCenter(hand_inertia_simbody, translation_hand_comArm_inRadiusFrame, hand_mass)

# 4. Rotate inertias according to the orientation of the humerus frame 
# (in this way, their sum can be assigned as a cumulative inertia of the system to the humerus alone)
rotation_ulnaFrameToHumFrame = body_ulna.findTransformBetween(state, body_humerus).R().toMat33()
rotation_radiusFrameToHumFrame = body_radius.findTransformBetween(state, body_humerus).R().toMat33()
rotation_handFrameToHumFrame = body_hand.findTransformBetween(state, body_humerus).R().toMat33()

ulna_inertia_translated_inHumFrame = utils.rotateInertia(ulna_inertia_translated_inUlnaFrame, rotation_ulnaFrameToHumFrame)
radius_inertia_translated_inHumFrame = utils.rotateInertia(radius_inertia_translated_inRadiusFrame, rotation_radiusFrameToHumFrame)
hand_inertia_translated_inHumFrame = utils.rotateInertia(hand_inertia_translated_inHandFrame, rotation_handFrameToHumFrame)

# this is the constant inertia in body-fixed frame
arm_inertia_inHumFrame = utils.sumInertias(hum_inertia_translated_inHumFrame, utils.sumInertias(hand_inertia_translated_inHumFrame, utils.sumInertias(ulna_inertia_translated_inHumFrame, radius_inertia_translated_inHumFrame)))

# Now, print all of the parameters that we have computed to screen

print("Center of mass of the full arm [in humerus frame]:")
print(arm_com_locationInHum_vec3.to_numpy())

print("\n\nTotal mass of the full arm:")
print(arm_mass)

print("\n\nTotal inertia of the full arm [in humerus frame]:")
print(utils.fromInertiaToNumpyArray(arm_inertia_inHumFrame))

# Generate a new OpenSim model that has the same body structure as model_simplified, but 
# with the same inertial properties of the complete one for what concerns the arm (with elbow bent at 90 degrees).
# The resulting model is model_simplified_updated, and its equivalence with the original one can be tested in
# test_inertias.py

# initialize simplified model and get its state
state_simpl = model_simplified.initSystem()

# get the humerus body of the simplified model, and modify its inertia properties
body_humerus_simpl = model_simplified.updBodySet().get('humerus')
body_humerus_simpl.setMassCenter(arm_com_locationInHum_vec3)
body_humerus_simpl.setMass(arm_mass)
body_humerus_simpl.setInertia(arm_inertia_inHumFrame)

# save the modified model (in the main directory of the repository)
model_new = model_simplified.clone()
model_new.setName('Model_simplified_with_full_inertia')
model_new.printToXML(model_new_name)

## TESTING the new model
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
modelProcessor_new = osim.ModelProcessor(os.path.join(path_to_repo, model_new_name))
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
