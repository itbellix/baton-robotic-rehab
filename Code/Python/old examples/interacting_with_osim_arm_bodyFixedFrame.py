
# In this script, I want to learn how to apply a give force to a muscle-less OpenSim model.
# The goal is to use the thoracoscapular shoulder model (with all coordinates locked apart from plane of elevation, shoulder elevation and scapula winging).
# In this script I also want to get the equivalent inertia of the arm, so that I can identify the inertia parameters of the arm directly from the model and 
# use them to obtain meaningful system dynamics to employ in my trajectory optimization.

# NOTE: working on simplified CustomJoint for the glenohumeral mobility
# TODO: clarify how to express accelerations analytically!

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import os
import utilities_from_Simbody as utils


# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..\..')             # getting path to the repository
path_to_model = os.path.join(path_to_repo, 'OsimModels')    # getting path to the OpenSim models

# load OpenSim model
with_gimbal = False
if with_gimbal:
    model = osim.Model(os.path.join(path_to_model, 'ScapulothoracicJoint_arm_locked_GimbalJoint.osim'))         # inertia is identified correctly
else:
    model = osim.Model(os.path.join(path_to_model, 'ScapulothoracicJoint_arm_locked.osim'))                     # inertia is still incorrect

# retrieve the bodies composing the arm and select the point of interest in the elbow
body_humerus = model.updBodySet().get('humerus')
body_ulna = model.updBodySet().get('ulna')
body_radius = model.updBodySet().get('radius')
body_hand = model.updBodySet().get('hand')

elbow_center = model.getMarkerSet().get('centelbow')

# initialize the model and get state
state = model.initSystem()

# get the location of the elbow center in humerus frame
loc_elbow_center_humerus = elbow_center.findLocationInFrame(state, body_humerus).to_numpy()

# ------- FINDING THE INERTIA OF THE HUMAN ARM IN BODY FIXED FRAMES -----------------------------------------------------------------------------
# get the inertia of the arm (humerus+ulna+radius) in the current configuration (elbow locked at 90 degree) and express it in the frame
# in which our shoulder coordinates are defined (i.e., the scapula_offset frame)
#   0. find the glenohumeral frame and its properties
#   1. retrieve the inertias of the bodies, expressed in their reference frames
#   2. "translate" the inertia matrices to the the same point (center of the GH joint)
#   3. "rotate" the inertia matrices to be all expressed in the same frame (scapula_offset)
#
# 0.
gh_base_frame = model.getJointSet().get('glenohumeral').get_frames(1)           # this is selecting the humerus_offset frame (child frame of GH joint)
gh_base_frame_locationInGround = gh_base_frame.getPositionInGround(state).to_numpy()
ground_frame = model.getJointSet().get('ground_thorax').get_frames(0)

# 1.  Note that inertias are expressed about the mass center, not the origin of the body frames!
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

# 2.
humerus_com_locationInGround = body_humerus.findStationLocationInGround(state, body_humerus.getMassCenter()).to_numpy()     # find the locations of the body CoMs (in ground)
ulna_com_locationInGround = body_ulna.findStationLocationInGround(state, body_ulna.getMassCenter()).to_numpy()
radius_com_locationInGround = body_radius.findStationLocationInGround(state, body_radius.getMassCenter()).to_numpy()
hand_com_locationInGround = body_hand.findStationLocationInGround(state, body_hand.getMassCenter()).to_numpy()

distance_hum_ghCenter = gh_base_frame_locationInGround - humerus_com_locationInGround        # compute the distance from each body frame and the gh_frame (in ground)
distance_hum_ghCenter_vec3 = osim.Vec3(distance_hum_ghCenter[0], distance_hum_ghCenter[1], distance_hum_ghCenter[2])

distance_ulna_ghCenter = gh_base_frame_locationInGround - ulna_com_locationInGround
distance_ulna_ghCenter_vec3 = osim.Vec3(distance_ulna_ghCenter[0], distance_ulna_ghCenter[1], distance_ulna_ghCenter[2])

distance_radius_ghCenter = gh_base_frame_locationInGround - radius_com_locationInGround
distance_radius_ghCenter_vec3 = osim.Vec3(distance_radius_ghCenter[0], distance_radius_ghCenter[1], distance_radius_ghCenter[2])

distance_hand_ghCenter = gh_base_frame_locationInGround - hand_com_locationInGround
distance_hand_ghCenter_vec3 = osim.Vec3(distance_hand_ghCenter[0], distance_hand_ghCenter[1], distance_hand_ghCenter[2])

translation_hum_gh_inHumFrame = ground_frame.expressVectorInAnotherFrame(state, distance_hum_ghCenter_vec3, body_humerus).to_numpy()   # express the distances in the correct body frames
translation_ulna_gh_inUlnaFrame = ground_frame.expressVectorInAnotherFrame(state, distance_ulna_ghCenter_vec3, body_ulna).to_numpy()
translation_radius_gh_inRadiusFrame = ground_frame.expressVectorInAnotherFrame(state, distance_radius_ghCenter_vec3, body_radius).to_numpy()
translation_hand_gh_inRadiusFrame = ground_frame.expressVectorInAnotherFrame(state, distance_hand_ghCenter_vec3, body_hand).to_numpy()

humerus_inertia_translated_inHumFrame = utils.shiftFromMassCenter(humerus_inertia_simbody, translation_hum_gh_inHumFrame, humerus_mass) # update inertias to be all expressed at the same point - center of GH frame (scapula_offset) - but still body frames
ulna_inertia_translated_inUlnaFrame = utils.shiftFromMassCenter(ulna_inertia_simbody, translation_ulna_gh_inUlnaFrame, ulna_mass)
radius_inertia_translated_inRadiusFrame = utils.shiftFromMassCenter(radius_inertia_simbody, translation_radius_gh_inRadiusFrame, radius_mass)
hand_inertia_translated_inHandFrame = utils.shiftFromMassCenter(hand_inertia_simbody, translation_hand_gh_inRadiusFrame, hand_mass)

# 3.
rotation_humFrameToGhFrame = body_humerus.findTransformBetween(state, gh_base_frame).R().toMat33()    # find a rotation from each body frame to gh_base frame
rotation_ulnaFrameToGhFrame = body_ulna.findTransformBetween(state, gh_base_frame).R().toMat33()
rotation_radiusFrameToGhFrame = body_radius.findTransformBetween(state, gh_base_frame).R().toMat33()
rotation_handFrameToGhFrame = body_hand.findTransformBetween(state, gh_base_frame).R().toMat33()

humerus_inertia_translated_inGhFrame = utils.rotateInertia(humerus_inertia_translated_inHumFrame, rotation_humFrameToGhFrame)
ulna_inertia_translated_inGhFrame = utils.rotateInertia(ulna_inertia_translated_inUlnaFrame, rotation_ulnaFrameToGhFrame)
radius_inertia_translated_inGhFrame = utils.rotateInertia(radius_inertia_translated_inRadiusFrame, rotation_radiusFrameToGhFrame)
hand_inertia_translated_inGhFrame = utils.rotateInertia(hand_inertia_translated_inHandFrame, rotation_handFrameToGhFrame)

arm_inertia_inGhFrame = utils.sumInertias(hand_inertia_translated_inGhFrame, utils.sumInertias(humerus_inertia_translated_inGhFrame, utils.sumInertias(ulna_inertia_translated_inGhFrame, radius_inertia_translated_inGhFrame)))
arm_inertia_inGhFrame_np = utils.fromInertiaToNumpyArray(arm_inertia_inGhFrame)     # this is the constant inertia in body-fixed frame

# ------------------- INDUCING ACCELERATIONS ON THE ARM WITH EXTERNAL FORCE ------------------------------------------------------------------
# create a PrescribedForce object for applying a known force to the model at the elbow
pf_elbow = osim.PrescribedForce("elbow_force", body_humerus)
pf_elbow.setPointIsInGlobalFrame(False)         # I want the application point to be expressed in the body frame
pf_elbow.setForceIsInGlobalFrame(False)          # I want the force components to be expressed in the global frame
pf_elbow.setPointFunctions(osim.Constant(loc_elbow_center_humerus[0]), osim.Constant(loc_elbow_center_humerus[1]), osim.Constant(loc_elbow_center_humerus[2]))
pf_elbow.setForceFunctions(osim.Constant(0), osim.Constant(0), osim.Constant(0))    # creating the field in advance, to access it later
pf_elbow.setTorqueFunctions(osim.Constant(0), osim.Constant(0), osim.Constant(0))   # creating the field in advance, to access it later

# add the PrescribedForce to the model
model.addForce(pf_elbow)

# obtain a writable component to update the value of the torque on the fly
forceset = pf_elbow.updForceFunctions()
torqueset = pf_elbow.updTorqueFunctions()

force_x = osim.Constant.safeDownCast(forceset.get(0))
force_y = osim.Constant.safeDownCast(forceset.get(1))
force_z = osim.Constant.safeDownCast(forceset.get(2))

torque_x = osim.Constant.safeDownCast(torqueset.get(0))
torque_y = osim.Constant.safeDownCast(torqueset.get(1))
torque_z = osim.Constant.safeDownCast(torqueset.get(2))

# finalize model
model.finalizeConnections()
state = model.initSystem()


## apply a time-varying force to the model at the elbow, and retrieve the accelerations that are produced.
## The force to be applied must be expressed in the gh_base_frame, and transformed to the `body_humerus` frame before application

# preallocate 2 different force/torque vectors, to rule out the effect of gravity on the DoF's accelerations
ft_1 = np.array([10, 4, -7, 0, 0, 0]).reshape((1,6))
ft_to_apply = np.vstack((np.zeros((1,6)), ft_1))

# preallocate vectors to save the accelerations produced by the forces (order will be plane of elevation, shoulder elevation, axial rotation)
accelerations = np.zeros((ft_to_apply.shape[0], 3))

# retrieve references to the coordinates I am interested in
pe = model.getCoordinateSet().get('plane_of_elev')
se = model.getCoordinateSet().get('shoulder_elv')
ar = model.getCoordinateSet().get('axial_rot')

delta_t = 0.01

for index in range(ft_to_apply.shape[0]):
    # set state time, to allow for setting different values of the force functions
    state.setTime(index*delta_t)

    # apply the components of the force/torque to the model
    force_x.setValue(ft_to_apply[index,0])
    force_y.setValue(ft_to_apply[index,1])
    force_z.setValue(ft_to_apply[index,2])
    torque_x.setValue(ft_to_apply[index,3])
    torque_y.setValue(ft_to_apply[index,4])
    torque_z.setValue(ft_to_apply[index,5])

    # update the state of the model and realize it to the acceleration stage
    model.assemble(state)
    model.equilibrateMuscles(state)
    model.realizeAcceleration(state)

    accelerations[index, :] = [se.getAccelerationValue(state), ar.getAccelerationValue(state), pe.getAccelerationValue(state)]

delta_acc = accelerations[1, :] - accelerations[0, :]

# check if the difference in the accelerations (without considering the gravitational effects) is explained correctly by the relation
# T = I*omega_dot + omega x (I*omega) (Euler equations for the rigid body, where T: torque expressed in body-fixed frame
#                                                                                omega: angular velocity vector in body-fixed frame
#                                                                                I: inertia matrix in body-fixed frame
pe_val = pe.getValue(state)
se_val = se.getValue(state)
ar_val = ar.getValue(state)

pe_dot = pe.getSpeedValue(state)
se_dot = se.getSpeedValue(state)
ar_dot = ar.getSpeedValue(state)
omega = np.array([se_dot, ar_dot, pe_dot])

Rz = np.array([np.cos(pe_val), -np.sin(pe_val), 0, 
               np.sin(pe_val), np.cos(pe_val), 0,
               0, 0, 1]).reshape((3,3))

Ry = np.array([np.cos(ar_val), 0, np.sin(ar_val),
               0, 1, 0, 
               -np.sin(ar_val), 0, np.cos(ar_val)]).reshape((3,3))

Rx = np.array([1, 0, 0,
               0, np.cos(se_val), -np.sin(se_val),
               0, np.sin(se_val), np.cos(se_val)]).reshape((3,3))

R = np.matmul(Rx, np.matmul(Ry, Rz))   # this is the rotation matrix transforming points from inertial to body-fixed frame

cross_prod = np.cross(omega, np.matmul(arm_inertia_inGhFrame_np, omega))
torque_inHumFrame = np.cross(loc_elbow_center_humerus, ft_1[0, 0:3])
omega_dot = np.matmul(np.linalg.inv(arm_inertia_inGhFrame_np), (torque_inHumFrame - cross_prod).T)

print(delta_acc)
print(omega_dot.reshape((3,)))

# print(R)
# print(arm_inertia_inGhFrame_np)


