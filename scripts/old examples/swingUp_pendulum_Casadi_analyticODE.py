# In this script, I want to test the performances of solving a 1DoF pendulum swing-up with CasADi, 
# by providing the algorithm with the analitical expression for the system's ODE.

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for optimal control
T = 3.   # Time horizon
N = 30  # number of control intervals
h = T / N

# Degree of interpolating polynomial
d = 3

# define actuation limits
max_torque = 20 # Nm

# Get collocation points
tau = ca.collocation_points(d, 'legendre')

# Collocation linear maps
C, D, B = ca.collocation_coeff(tau)

# Declare model variables
x = ca.MX.sym('x', 2)  # state vector: angular position and velocity of the pendulum [rad, rad/s] (x[0] = 0 means the pendulum is at its stable equilibrium point)

u = ca.MX.sym('u')


x_goal = np.array([np.pi, 0]) # desired state at the end of the horizon - pendulum straight up
x_0 = np.array([0, 0])        # initial condition - pendulum down

# Model equations
mass = 1
length = 1
inertia = mass*length**2
g = 9.81

xdot = ca.vertcat(x[1], u / (inertia) - g/length * ca.sin(x[0]))

# Objective term
L = (x[0] - x_goal[0])**2 + u**2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L])

# Start with an empty NLP
opti = ca.Opti()
J = 0

# "Lift" initial conditions
Xk = opti.variable(2)
x_init = opti.parameter(2)
opti.subject_to(Xk == x_init)
opti.set_value(x_init, x_0)

# Collect all states/controls
Xs = [Xk]
Us = []

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = opti.variable()
    Us.append(Uk)
    opti.subject_to(-max_torque <= Uk)
    opti.subject_to(Uk <= max_torque)

    # Decision variables for helper states at each collocation point
    Xc = opti.variable(2, d)

    # Evaluate ODE right-hand-side at all helper states
    ode, quad = f(Xc, Uk)

    # Add contribution to quadrature function
    J = J + h*ca.mtimes(quad, B)

    # Get interpolating points of collocation polynomial
    Z = ca.horzcat(Xk, Xc)

    # Get slope of interpolating polynomial (normalized)
    Pidot = ca.mtimes(Z, C)
    # Match with ODE right-hand-side
    opti.subject_to(Pidot == h * ode)

    # State at the end of collocation interval
    Xk_end = ca.mtimes(Z, D)

    # New decision variable for state at the end of the interval
    Xk = opti.variable(2)
    Xs.append(Xk)

    # Continuity constraints
    opti.subject_to(Xk_end == Xk)

# adding constraint to reach the final desired state
opti.subject_to(Xk==x_goal)

# Flatten lists
Xs = ca.vertcat(*Xs)
Us = ca.vertcat(*Us)

opti.minimize(J)

opts = {'ipopt.print_level': 3,
        'print_time': 3,
        'ipopt.tol': 1e-3,
        'expand': 1,
        'error_on_fail': 1,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.warm_start_bound_push': 1e-8,
        'ipopt.warm_start_mult_bound_push': 1e-8,
        'ipopt.mu_init': 1e-5,
        'ipopt.bound_relax_factor': 1e-9}

opti.solver('ipopt', opts)

sol = opti.solve()

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
