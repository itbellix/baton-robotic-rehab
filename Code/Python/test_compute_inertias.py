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
import utilities_from_Simbody_copy as utils
    
## PARAMS -------------------------------------------------------------------------------------------------------
model_complete_name = 'debug_2bodies.osim'      # name of the full model that has all the required bodies
model_simplified_name = 'debug_1body.osim'      # name of the model with only one moving body
model_new_name = 'debug_1body_full.osim'        # name of the model that will be produced 
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
body1 = model_complete.updBodySet().get('body1')
body2 = model_complete.updBodySet().get('body2')

# initialize the complete model to work on it, and get its state
state = model_complete.initSystem()

# ------- FINDING THE INERTIA OF THE L-SHAPED BODY in the required configuration -----------------------------------------------------------------------------
ground_frame = model_complete.get_ground()

# 1.  Note that inertias are expressed about the mass center, not the origin of the body frames!
#     Also, each inertia is expressed in a different (body-fixed) frame
body1_inertia = utils.fromVec6InertiatoNumpyArray(body1.get_inertia())
body1_mass = body1.getMass()

body2_inertia = utils.fromVec6InertiatoNumpyArray(body2.get_inertia())
body2_mass = body2.getMass()

total_mass = body1_mass + body2_mass

# 2. Find the position of the CoM of L-shaped object
# find the locations of the body CoMs (in ground)
body1_com_locationInGround = body1.findStationLocationInGround(state, body1.getMassCenter()).to_numpy()
body2_com_locationInGround = body2.findStationLocationInGround(state, body2.getMassCenter()).to_numpy()

total_com_locationInGround = (body1_mass*body1_com_locationInGround + body2_mass*body2_com_locationInGround) / total_mass

# find the location of the new center of mass, in the body1 reference frame
total_com_locationInGround_vec3 = osim.Vec3(total_com_locationInGround[0], total_com_locationInGround[1], total_com_locationInGround[2])
total_com_locationInBody1_vec3 = ground_frame.findStationLocationInAnotherFrame(state, total_com_locationInGround_vec3, body1)

# 3. Translate all of the inertias to the center of mass of the overall multibody system
distance_body1_comTotal = total_com_locationInGround - body1_com_locationInGround
distance_body1_comTotal_vec3 = osim.Vec3(distance_body1_comTotal[0], distance_body1_comTotal[1], distance_body1_comTotal[2])

distance_body2_comTotal = total_com_locationInGround - body2_com_locationInGround
distance_body2_comTotal_vec3 = osim.Vec3(distance_body2_comTotal[0], distance_body2_comTotal[1], distance_body2_comTotal[2])

# express the distances in the correct body frames (TODO: this is probably useless, the distance is already in body frame!)
translation_body1_comTotal_inBody1Frame = ground_frame.expressVectorInAnotherFrame(state, distance_body1_comTotal_vec3, body1).to_numpy()
translation_body2_comTotal_inBody2Frame = ground_frame.expressVectorInAnotherFrame(state, distance_body2_comTotal_vec3, body2).to_numpy()

# update inertias to be all expressed at the same point - center of mass of the arm - but still body frames
body1_inertia_translated_inBody1Frame = utils.shiftFromMassCenter(body1_inertia, translation_body1_comTotal_inBody1Frame, body1_mass)
body2_inertia_translated_inBody2Frame = utils.shiftFromMassCenter(body2_inertia, translation_body2_comTotal_inBody2Frame, body2_mass)

# 4. Rotate inertias according to the orientation of the body1 frame 
# (in this way, their sum can be assigned as a cumulative inertia of the system to the body1 alone)
rotation_body1FrameToBody2Frame = utils.fromMat33toNumpyArray(body2.findTransformBetween(state, body1).R().toMat33())

# body2_inertia_translated_inBody1Frame = utils.rotateInertia(body2_inertia_translated_inBody2Frame, rotation_body1FrameToBody2Frame)
body2_inertia_translated_inBody1Frame = body2_inertia_translated_inBody2Frame   # as the orientation is the same

# this is the constant inertia in body-fixed frame
total_inertia_inBody1Frame = utils.sumInertias(body1_inertia_translated_inBody1Frame, body2_inertia_translated_inBody1Frame)

# Now, print all of the parameters that we have computed to screen

print("Center of mass of the full model [in body1 frame]:")
print(total_com_locationInBody1_vec3.to_numpy())

print("\n\nTotal mass of the full arm:")
print(total_mass)

print("\n\nTotal inertia of the full arm [in humerus frame]:")
print(total_inertia_inBody1Frame)

# Now we need to manually copy these results into the target model (model_simplified_name),
# and test the result in test_model_simplified.py
