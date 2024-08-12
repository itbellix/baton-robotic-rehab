"""
Here I collect the classes and functions that are needed to ease the use of an OpenSim module 
with CasADi in the trajectory optimization module.

Essentially, the OpenSim module is queried for the evolution of the state variables of our shoulder
model, capturing the dynamics of the skeletal system. This is implemented in a CasADi callback,
that CasADi can use at run time to construct gradient and hessian of the optimization problem numerically.
"""

import casadi as ca
import opensim as osim
import numpy as np

class MyOsimCallback(ca.Callback):
    """
    This class is derived from the callback class implemented by CasADi, and allows to calculate
    the state derivative of an OpenSim model. It essentially implements the function that maps from 
    {x,u} to x_ddot (aka, x_ddot=F(x,u)), where 
    x: system state, in terms of coordinate values and speeds
    u: control input, in terms of generalized forces
    """
    def __init__(self, name, osim_model, opts={}):
        ca.Callback.__init__(self)
        self.model = osim_model
        self.state = self.model.initSystem()
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoords = self.coordinateSet.getSize()
        self.nActs = self.model.getActuators().getSize()

        # prepare the actuators in the model to be overwritten
        # note that our model should feature no muscles, only CoordinateActuators
        self.acts = []
        for index_act in range(self.nActs):
            self.acts.append(osim.ScalarActuator.safeDownCast(self.model.getActuators().get(index_act)))
            if not(self.acts[index_act].isActuationOverridden(self.state)):
                self.acts[index_act].overrideActuation(self.state, True)

        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 2*self.nCoords + self.nActs   # the idea here is that the inputs are the coordinate values and speeds (collectively forming x) plus controls
    def get_n_out(self): return 2*self.nCoords               # outputs are x_dot: coordinate velocities and accelerations (i.e. the time derivative of x)

    # Initialize the object
    def init(self):
      print('initializing object')

  # Evaluate numerically
    def eval(self, args):
      # distinguish between coordinate values, speeds and controls provided to the callback
      state_vars = args[0:-self.nActs]
      controls = args[-self.nActs:]
      coord_vals = state_vars[0::2]
      coord_speeds = state_vars[1::2]

      # transform state variables into a SimTk vector
      state_vars_vec = osim.Vector(6, 0)
      for var in range(len(state_vars)):
        state_vars_vec[var] = state_vars[var].full()[0][0]

      # set the values of the state variables in the model
      self.model.setStateVariableValues(self.state, state_vars_vec)

      # set the actuation level for each coordinate actuators
      for act in range(self.nActs):
        self.acts[act].setOverrideActuation(self.state, controls[act].full()[0][0])

      # initialize the model and realize to acceleration stage
      self.model.realizeAcceleration(self.state)
      
      # retrieve accelerations for the single coordinates
      accelerations = np.zeros((self.nCoords,))

      # get the accelerations for each coordinate
      for coor in range(self.nCoords):
        accelerations[coor] = self.model.getCoordinateSet().get(coor).getAccelerationValue(self.state)

      return [coord_speeds[0].full()[0],    # theta_dot
              accelerations[0],             # theta_ddot
              coord_speeds[1].full()[0],    # psi_dot
              accelerations[1],             # psi_ddot
              coord_speeds[2].full()[0],    # phi_dot (unused in our case)            
              accelerations[2]]             # phi_ddot (unused in our case)
    