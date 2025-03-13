# In this script, I want to test the performances of solving a 1DoF pendulum swing-up with CasADi, 
# by providing the algorithm with a numerical way to compute the system's ODE though CasADi callbacks containing OpenSim functions

import casadi as ca
import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile, pstats, io
from pstats import SortKey

withviz = False

# ----------- FIRST, some testing with the Callback in CasADi + OpenSim -------------------------------
# 1. Set up my own callback leveraging OpenSim
class MyOsimCallback(ca.Callback):
  def __init__(self, name, osim_model, state, nCoords, opts={}):
    ca.Callback.__init__(self)
    self.model = osim_model
    self.coordinateSet = self.model.getCoordinateSet()
    self.state = state
    self.nCoords = nCoords
    self.nActs = self.model.getActuators().getSize()

    self.acts = []
    for index_act in range(self.nActs):
      self.acts.append(osim.ScalarActuator.safeDownCast(self.model.getActuators().get(index_act)))
      if not(self.acts[index_act].isActuationOverridden(state)):
        self.acts[index_act].overrideActuation(state, True)

    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 2*nCoords + self.nActs   # the idea here is that the inputs are the coordinate values and speeds (collectively forming x) plus controls
  def get_n_out(self): return 2*nCoords               # outputs are x_dot: coordinate velocities and accelerations (i.e. the time derivative of x)

  # Initialize the object
  def init(self):
     print('initializing object')

  # Evaluate numerically
  def eval(self, args):
    # distinguish between coordinate values, speeds and controls provided to the callback
    coord_vals = args[0:self.nCoords]
    coord_speeds = args[self.nCoords:-self.nActs]
    controls = args[-self.nActs:]

    # set the values of the state variables in the model
    #TODO: for loop (could be slowing things down, try with setStateVariableValues)
    for coor in range(self.nCoords):
      self.coordinateSet.get(coor).setValue(self.state, coord_vals[coor].full()[0][0])
      self.coordinateSet.get(coor).setSpeedValue(self.state, coord_speeds[coor].full()[0][0])

    # TODO: for loop (could be slowing things down)
    for act in range(self.nActs):
      self.acts[act].setOverrideActuation(self.state, controls[act].full()[0][0])

    # initialize the model and realize to acceleration stage
    self.model.realizeAcceleration(self.state)
    
    # retrieve accelerations for the single coordinates
    accelerations = np.zeros((1, nCoords))
    #TODO: for loop (could be slowing things down, try with getUdots)
    for coor in range(self.nCoords):
      accelerations[coor] = self.model.getCoordinateSet().get(coor).getAccelerationValue(self.state)

    return [coord_speeds[0].full()[0], accelerations[0]]

# Use the function
model = osim.Model('/home/itbellix/Desktop/GitHub/PTbot_official/OsimModels/simple_pendulum.osim')
state = model.initSystem()
coordinateSet = model.getCoordinateSet()
nCoords = coordinateSet.getSize()
coordinates = []
for coor in range(nCoords):
  coordinates.append(coordinateSet.get(coor).getName())

sys_dynamics = MyOsimCallback('sys_dynamics', model, state, nCoords, {"enable_fd":True})
res = sys_dynamics(0.2, 1.2, 1.0)
print(res)


# You may call the Callback symbolically
pos = ca.MX.sym('pos')
vel = ca.MX.sym('vel')
tau = ca.MX.sym('tau')
print(sys_dynamics(pos, vel, tau))

# -------- THEN, we can set up the optimal control problem -------------------------------
# pr = cProfile.Profile()
# pr.enable()

# Define parameters for optimal control -------------------------------------------------------------------------------
T = 3.     # Time horizon
N = 30      # number of control intervals
h = T/N

# Degree of interpolating polynomial
pol_order = 3

# Define actuation limits
max_torque = 20 #Nm

# Get collocation points
tau = ca.collocation_points(pol_order, 'legendre')

# Get linear maps
C, D, B = ca.collocation_coeff(tau)

# Declare model variables
x = ca.MX.sym('x', 2)       # state vector: angular position and velocity of the pendulum [rad, rad/s] (x[0] = 0 means the pendulum is at its stable equilibrium point)

u = ca.MX.sym('u')          # control vector [Nm]

x_goal = np.array([np.pi, 0]) # desired state at the end of the horizon - pendulum straight up
x_0 = np.array([0, 0])        # initial condition - pendulum down

# Cost terms ---- the cost function will minimize the energy consumption while attaining the desired final position 
L = (x[0] - x_goal[0])**2 + u**2
cost_function = ca.Function('cost_function', [x, u], [L])

# MPC loop -----------------------------

opti = ca.Opti()        # optimization problem
J = 0


#  "Lift" initial conditions
Xk = opti.variable(2)
x_init = opti.parameter(2)
opti.subject_to(Xk==x_init)
opti.set_value(x_init, x_0)


# Collect all states/controls
Xs = [Xk]
Us = []

# formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = opti.variable()
    Us.append(Uk)
    opti.subject_to(-max_torque <= Uk)
    opti.subject_to(Uk <= max_torque)

    # optimization variable (state) at collocation points
    Xc = opti.variable(2, pol_order)

    # evaluate ODE right-hand-side at collocation points (TODO: this depends from the number of coordinates!)
    ode_tuple = sys_dynamics(Xc[0,:], Xc[1, :], Uk) # this used to be Xk but was wrong. How to do it correctly?
    ode_MX = ca.vertcat(ode_tuple[0], ode_tuple[1])

    quad = cost_function(Xc, Uk)

    # add contribution to quadrature function
    J = J + h*ca.mtimes(quad,B)

    # get interpolating points of collocation polynomial
    Z = ca.horzcat(Xk, Xc)

    # get slope of interpolating polynomial (normalized)
    Pidot = ca.mtimes(Z, C)
    # match with ODE right-hand-side
    opti.subject_to(Pidot==h*ode_MX)

    # state at the end of collocation interval
    Xk_end = ca.mtimes(Z, D)

    # new decision variable for state at the end of interval
    Xk = opti.variable(2)
    Xs.append(Xk)
    
    # continuity constraint
    opti.subject_to(Xk_end==Xk)

# adding constraint to reach the final desired state
opti.subject_to(Xk==x_goal)

Us = ca.vertcat(*Us)
Xs = ca.vertcat(*Xs)

# define the cost function to be minimized
opti.minimize(J)

# choose the solver
opts = {'ipopt.print_level': 3,
        'print_time': 3,
        'ipopt.tol': 1e-3,
        'expand': 0,
        'error_on_fail': 1,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.warm_start_bound_push': 1e-8,
        'ipopt.warm_start_mult_bound_push': 1e-8,
        'ipopt.mu_init': 1e-5,
        'ipopt.bound_relax_factor': 1e-9,
        'ipopt.hessian_approximation': 'limited-memory'}

opti.solver('ipopt', opts)

# solve the NLP problem
sol = opti.solve()

# retrieve the optimal values
x_opt = sol.value(Xs)
u_opt = sol.value(Us)
lam_g0 = sol.value(opti.lam_g)

# convert to casadi function adn solve the problem again with initial guess
F = opti.to_function('F', [x_init, Xs, Us, opti.lam_g], [Xs, Us, opti.lam_g])
x_opt1, u_opt1, lam_g0 = F(x_0, x_opt, u_opt, lam_g0)
x_opt1 = x_opt1.full()
u_opt1 = u_opt1.full()

# distinguish between angles and velocities
theta_opt = x_opt[::2]
theta_dot_opt = x_opt[1::2]
theta_opt1 = x_opt1[::2]
theta_dot_opt1 = x_opt1[1::2]

# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())

# apply the values with Forward Dynamics and visualize the effect with the OpenSim visualizer (just to be sure that they are correct)------------------------------
if withviz:
  model.setUseVisualizer(True)
  state = model.initSystem()

# set the coordinate in the initial position
for coord in range(nCoords):
  model.getCoordinateSet().get(coord).setValue(state, theta_opt[0])
  model.getCoordinateSet().get(coord).setSpeedValue(state, theta_dot_opt[0])

# retrieve actuators and prepare them for being overridden
nActs = model.getActuators().getSize()
acts = []
for index_act in range(nActs):
    acts.append(osim.ScalarActuator.safeDownCast(model.getActuators().get(index_act)))
    if not(acts[index_act].isActuationOverridden(state)):
      acts[index_act].overrideActuation(state, True)


# using OpenSim Manager
theta_fd = [theta_opt[0]]
theta_dot_fd = [theta_dot_opt[0]]
for k in range(N):

  # command the optimized torque to the model
  for index_act in range(nActs):
    acts[index_act].setOverrideActuation(state, u_opt[k])

  # instatiate the manager and use it to integrate the system  
  manager = osim.Manager(model, state)
  state = manager.integrate(h+k*h)

  # retrieve the values of the angles and angular velocities
  theta_fd.append(model.getCoordinateSet().get(0).getValue(state))
  theta_dot_fd.append(model.getCoordinateSet().get(0).getSpeedValue(state))


# Plot the solution
tgrid = np.linspace(0, T, N + 1)
plt.figure()
plt.plot(tgrid, theta_opt[:], '--')
plt.plot(tgrid, theta_dot_opt[:], '-')
plt.plot(tgrid, theta_opt1[:], '--')
plt.plot(tgrid, theta_dot_opt1[:], '-')
plt.xlabel('t')
plt.ylabel('[rad] , [rad/s]')
plt.legend(['theta [rad]', 'theta_dot [rad/s]', 'theta IC [rad]', 'theta_dot IC [rad/s]'])

plt.figure()
plt.stairs(u_opt)
plt.stairs(u_opt1[:,0])
plt.xlabel('control intervals')
plt.ylabel('[Nm]')
plt.legend(['control torque', 'control torque IC'])
plt.show()


# plot using OpenSim visualizer
if withviz:
  for time_instant in range(N):
    for coor in range(nCoords):
      coordinateSet.get(coor).setValue(state, theta_opt[time_instant])

    model.getVisualizer().show(state)
    time.sleep(h)     # reproduce the motion in real time (same step used in simulation)


# 2. Option two would be to explore and modify an existing Callback example (from callback.py example @ CasADi).
# There are many ways in which we can provide some derivatives to CasADi, listed below. Could be worth exploring some.
# Derivates OPTION 1: finite-differences
# Derivates OPTION 2: Supply forward mode
# Derivates OPTION 3: Supply reverse mode
# Derivates OPTION 4: Supply full Jacobian
