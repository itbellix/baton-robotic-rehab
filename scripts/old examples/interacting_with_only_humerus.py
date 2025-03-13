# In this script, I want to learn how to apply a give force to a muscle-less OpenSim model.
# The goal is to use a simple pendulum model (only the humerus is present).
# In this script I also want to get the equivalent inertia of the humerus, so that I can identify the inertia parameters of the humerus directly from the model.

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import os
import utilities as utils


# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..\..')             # getting path to the repository
path_to_model = os.path.join(path_to_repo, 'OsimModels')    # getting path to the OpenSim models

# load OpenSim model
model = osim.Model(os.path.join(path_to_model, 'only_humerus.osim'))

# retrieve the bodies composing the arm and select the point of interest in the elbow
body_humerus = model.updBodySet().get('humerus')

elbow_center = model.getMarkerSet().get('centelbow')

# initialize the model and get state
state = model.initSystem()

# get the location of the elbow center in humerus frame
loc_elbow_center_humerus = elbow_center.findLocationInFrame(state, body_humerus).to_numpy()

# ------- FINDING THE INERTIA OF THE HUMERUS ----------------------------------------------------------------------------------------
# get the inertia of the humerus in the current configuration (elbow locked at 90 degree) and express it in the frame
# in which our shoulder coordinates are defined (i.e., the scapula_offset frame)
#   0. find the glenohumeral frame and its properties
#   1. retrieve the inertia of the body, expressed in its reference frames
#   2. "translate" the inertia matrix to the the joint center (of the GH joint)
#   3. "rotate" the inertia matrix to express it in the scapula_offset
#
# 0.
gh_base_frame = model.getJointSet().get('glenohumeral').get_frames(0)
gh_base_frame_locationInGround = gh_base_frame.getPositionInGround(state).to_numpy()
ground_frame = model.getJointSet().get('ground_thorax').get_frames(0)

# 1.
humerus_inertia = body_humerus.get_inertia()  # the inertia matrix is expressed about the mass center, not the body origin!
humerus_inertia_simbody = osim.simbody.Inertia(humerus_inertia.get(0), humerus_inertia.get(1), humerus_inertia.get(2), humerus_inertia.get(3), humerus_inertia.get(4), humerus_inertia.get(5))
humerus_mass = body_humerus.getMass()

# 2.
humerus_com_locationInGround = body_humerus.findStationLocationInGround(state, body_humerus.getMassCenter()).to_numpy()     # find the locations of the body CoM (in ground)

distance_hum_ghCenter = gh_base_frame_locationInGround - humerus_com_locationInGround       # compute the distance from body frame to the gh_frame (in ground)
distance_hum_ghCenter_vec3 = osim.Vec3(distance_hum_ghCenter[0], distance_hum_ghCenter[1], distance_hum_ghCenter[2])

translation_hum_gh_inHumFrame = ground_frame.expressVectorInAnotherFrame(state, distance_hum_ghCenter_vec3, body_humerus).to_numpy()   # express the distances in the correct body frames

humerus_inertia_translated_inHumFrame = utils.shiftFromMassCenter(humerus_inertia_simbody, translation_hum_gh_inHumFrame, humerus_mass) # update inertias to be all expressed at the same point - center of GH frame (scapula_offset) - but still body frames

# 3.
rotation_humFrameToGhFrame = body_humerus.findTransformBetween(state, gh_base_frame).R().toMat33()    # find a rotation from each body frame to gh_base frame

humerus_inertia_translated_inGhFrame = utils.rotateInertia(humerus_inertia_translated_inHumFrame, rotation_humFrameToGhFrame)

arm_inertia_inGhFrame = humerus_inertia_translated_inGhFrame
arm_inertia_inGhFrame_np = utils.fromInertiaToNumpyArray(arm_inertia_inGhFrame)

# ------------------- INDUCING ACCELERATIONS ON THE ARM WITH EXTERNAL FORCE ------------------------------------------------------------------
# create a PrescribedForce object for applying a known force to the model at the elbow
pf_elbow = osim.PrescribedForce("elbow_force", body_humerus)
pf_elbow.setPointIsInGlobalFrame(False)         # I want the application point to be expressed in the body frame
pf_elbow.setForceIsInGlobalFrame(False)         # I want the force components to be expressed in the body frame
pf_elbow.setPointFunctions(osim.Constant(loc_elbow_center_humerus[0]), osim.Constant(loc_elbow_center_humerus[1]), osim.Constant(loc_elbow_center_humerus[2]))
pf_elbow.setForceFunctions(osim.Constant(0), osim.Constant(0), osim.Constant(0))    # creating the field in advance, to access it later
pf_elbow.setTorqueFunctions(osim.Constant(0), osim.Constant(0), osim.Constant(0))   # creating the field in advance, to access it later

# add the PrescribedForce to the model
model.addForce(pf_elbow)

# obtain a writable component to update the value of the force/torque on the fly
forceset = pf_elbow.updForceFunctions()
torqueset = pf_elbow.updTorqueFunctions()

force_xHum = osim.Constant.safeDownCast(forceset.get(0))
force_yHum = osim.Constant.safeDownCast(forceset.get(1))
force_zHum = osim.Constant.safeDownCast(forceset.get(2))

torque_xHum = osim.Constant.safeDownCast(torqueset.get(0))
torque_yHum = osim.Constant.safeDownCast(torqueset.get(1))
torque_zHum = osim.Constant.safeDownCast(torqueset.get(2))

# finalize model
model.finalizeConnections()
state = model.initSystem()


## apply a time-varying force to the model at the elbow, and retrieve the accelerations that are produced.
## The force to be applied must be expressed in the gh_base_frame, and transformed to the `body_humerus` frame before application

# preallocate 2 different force vectors, to rule out the effect of gravity on the DoF's accelerations
torques_to_apply = np.vstack((np.zeros((1,3)), np.array([2, 2, 1]).reshape((1,3))))
# torques_to_apply = np.vstack((np.array([2, 2, 1]).reshape((1,3)), (np.zeros((1,3)))))
# torques_to_apply = np.array([2, 2, 1]).reshape((1,3))

# preallocate vectors to save the accelerations produced by the forces (order will be plane of elevation, shoulder elevation, axial rotation)
accelerations = np.zeros((torques_to_apply.shape[0], 3))

# retrieve references to the coordinates I am interested in
rx = model.getCoordinateSet().get('rx')
ry = model.getCoordinateSet().get('ry')
rz = model.getCoordinateSet().get('rz')

delta_t = 0.01

for index in range(torques_to_apply.shape[0]):
    # set state time, to allow for setting different values of the force functions
    state.setTime(index*delta_t)
    
    # select the force to be applied at the elbow, in the glenohumeral frame (`scapula_offset`)
    torques_inGhFrame = torques_to_apply[index, :]

    # transform it into humerus frame
    torques_inHumFrame = np.matmul(utils.fromMat33toNumpyArray(rotation_humFrameToGhFrame).transpose(), torques_inGhFrame)

    # apply the components of the force to the model
    torque_xHum.setValue(torques_inHumFrame[0])
    torque_yHum.setValue(torques_inHumFrame[1])
    torque_zHum.setValue(torques_inHumFrame[2])

    # update the state of the model and realize it to the acceleration stage
    model.assemble(state)
    model.equilibrateMuscles(state)
    model.realizeAcceleration(state)

    accelerations[index, :] = [rx.getAccelerationValue(state), ry.getAccelerationValue(state), rz.getAccelerationValue(state)]


delta_acc = accelerations[1, :] - accelerations[0, :]

# check if the difference in the accelerations (without considering the gravitational effects) is explained correctly by the relation
# T = I * omega_dot

omega_dot = np.matmul(np.linalg.inv(arm_inertia_inGhFrame_np), np.array([2, 2, 1]).reshape((3,1)))

print(delta_acc)
print(omega_dot.reshape((3,)))

