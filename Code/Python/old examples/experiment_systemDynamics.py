# In this script, I just want to make sure that I know what is the best way to retrieve the accelerations
# of the glenohumeral joint from the OpenSim shoulder model, given an external force

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import os

# define the required paths
code_path = os.path.dirname(os.path.realpath(__file__))     # getting path to where this script resides
path_to_repo = os.path.join(code_path, '..', '..')          # getting path to the repository
path_to_model = os.path.join(path_to_repo, 'OsimModels')    # getting path to the OpenSim models

# select model to work with
model = osim.Model(os.path.join(path_to_model, 'ScapulothoracicJoint_arm_locked.osim')) 

# initialize the model and get state
state = model.initSystem()

# retrieve body to apply the force to
force_application_body = model.getBodySet().get('ulna')

# retrieve marker to apply elbow force to 
force_application_point = model.getMarkerSet().get('centelbow').findLocationInFrame(state, force_application_body).to_numpy()

# retrieve references to the coordinates I am interested in
pe = model.getCoordinateSet().get('plane_of_elev')
se = model.getCoordinateSet().get('shoulder_elv')
ar = model.getCoordinateSet().get('axial_rot')

# create a PrescribedForce object for applying a known force to the model at the elbow
pf_elbow = osim.PrescribedForce("elbow_force", force_application_body)
pf_elbow.setPointIsInGlobalFrame(False)         # I want the application point to be expressed in the body frame
pf_elbow.setForceIsInGlobalFrame(False)         # I want the force components to be expressed in the body frame
pf_elbow.setPointFunctions(osim.Constant(force_application_point[0]), osim.Constant(force_application_point[1]), osim.Constant(force_application_point[2]))
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
