import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------- #
# 1. first, test with a simple function of which I know the analytical solution
# ------------------------------------------------------------------------------------------------------------------------- #

opt = ca.Opti()
theta = opt.variable()
psi = opt.variable()
param = opt.parameter()
opt.set_value(param, 0)

opt.minimize(0)
opt.subject_to(ca.cos(theta)*ca.sin(psi)==param)
# set solver options
opts = {'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.tol': 1e-3,
        'expand': 0,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.warm_start_bound_push': 1e-8,
        'ipopt.warm_start_mult_bound_push': 1e-8,
        'ipopt.mu_init': 1e-5,
        'ipopt.bound_relax_factor': 1e-9}

opt.solver('ipopt', opts)

# convert the opti object to a function, so that we can provide the initial guesses for the variables
# By changing the value of param in the function call, different local minima are found -> the initial guess is considered!
F_0 = opt.to_function('F_0', [param, theta, psi], [theta, psi], ['initial_state', 'guess_state_traj', 'guess_controls'], ['x_opt', 'u_opt'])
theta_opt, psi_opt = F_0(0, 10, 10)
print("Optimal solution is: theta = ", theta_opt, " , psi = ", psi_opt)

# ------------------------------------------------------------------------------------------------------------------------- #
# 2. then, test with the direct_collocation_opti.m example from https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
# The code, rewritten in Python, has been modified to parametrize the starting point x_0
# -------------------------------------------------------------------------------------------------------------------------- #

# Degree of interpolating polynomial
d = 3

# Get collocation points
tau = ca.collocation_points(d, 'legendre')

# Collocation linear maps
(C, D, B) = ca.collocation_coeff(tau)

# Time horizon
T = 10

# Declare model variables
x1 = ca.MX.sym('x1')
x2 = ca.MX.sym('x2')
x = ca.vertcat(x1, x2)
u = ca.MX.sym('u')

# Model equations
xdot = ca.vertcat((1 - x2**2) * x1 - x2 + u, x1)

# Objective term
L = x1**2 + x2**2 + u**2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L])

# Control discretization
N = 20  # number of control intervals
h = T / N

# Start with an empty NLP
opti = ca.Opti()
J = 0

# "Lift" initial conditions
Xk = opti.variable(2)
x_0 = opti.parameter(2)
opti.set_value(x_0, [0,1])
opti.subject_to(Xk == x_0)

# Collect all states/controls
Xs = [Xk]
Us = []

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = opti.variable()
    Us.append(Uk)
    opti.subject_to(-1 <= Uk)
    opti.subject_to(Uk <= 1)
    opti.set_initial(Uk, 0)

    # Decision variables for helper states at each collocation point
    Xc = opti.variable(2, d)
    opti.subject_to(-0.25 <= Xc[0, :])
    opti.set_initial(Xc, ca.repmat([0, 0], 1, d))

    # Evaluate ODE right-hand-side at all helper states
    ode, quad = f(Xc, Uk)

    # Add contribution to quadrature function
    J += ca.mtimes(quad, B) * h

    # Get interpolating points of collocation polynomial
    Z = ca.horzcat(Xk, Xc)

    # Get slope of interpolating polynomial (normalized)
    Pidot = ca.mtimes(Z, C)
    # Match with ODE right-hand-side
    opti.subject_to(Pidot == h * ode)

    # State at the end of the collocation interval
    Xk_end = ca.mtimes(Z, D)

    # New decision variable for the state at the end of the interval
    Xk = opti.variable(2)
    Xs.append(Xk)
    opti.subject_to(-0.25 <= Xk[0])
    opti.set_initial(Xk, [0, 0])

    # Continuity constraints
    opti.subject_to(Xk_end == Xk)

# Flatten lists of variables
Xs = ca.vertcat(*Xs)
Us = ca.vertcat(*Us)

opti.minimize(J)

# set solver options
opts = {'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.tol': 1e-3,
        'expand': 0,
        'error_on_fail':1,                              # to guarantee transparency if solver fails
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.warm_start_bound_push': 1e-8,
        'ipopt.warm_start_mult_bound_push': 1e-8,
        'ipopt.mu_init': 1e-5,
        'ipopt.bound_relax_factor': 1e-9}

opti.solver('ipopt', opts)

# Number of times to execute each part
n_times = 100  # You can change this to your desired number of executions

# Lists to store execution times for each part
execution_times_part1 = []
execution_times_part2 = []
execution_times_part3 = []
execution_times_part4 = []


# solve
print("\nSolving for the first time")
for _ in range(n_times):
    
    start = time.time()
    sol1 = opti.solve()
    execution_time = time.time() - start
    execution_times_part1.append(execution_time)
    lam_g0 = sol1.value(opti.lam_g)
    # print(sol1.stats()["iter_count"])

    x_opt = sol1.value(Xs)
    u_opt = sol1.value(Us)

# print("\nPassing initial makes a difference (with dual variables too)")
# opti.set_initial(sol1.value_variables())
# opti.set_initial(opti.lam_g, lam_g0)
# start = time.time()
# sol2 = opti.solve()
# print(time.time()-start)
# print(sol2.stats()["iter_count"])

# transform Opti stack to function
p = opti.parameter(2)
F_1 = opti.to_function('F_1', [p], [Xs, Us], ['initial_state'], ['x_opt', 'u_opt'])

# solve the same problem as above
print("\n\n\n\n Function with no initial guess")
for _ in range(n_times):
    x_0 = np.array([0, 1])
    start = time.time()
    x_opt_f1, u_opt_f1 = F_1(x_0)
    execution_time = time.time() - start
    execution_times_part2.append(execution_time)
    x_opt_f1 = x_opt_f1.full()
    u_opt_f1 = u_opt_f1.full()


# transform Opti stack to function (with initial guess provided for primal variable)
p = opti.parameter(2)
F_2 = opti.to_function('F_2', [p, Xs, Us], [Xs, Us], ['initial_state', 'guess_state_traj', 'guess_controls'], ['x_opt', 'u_opt'])

# solve the problem
print("\n\n\n\n Function with initial guess (primal variable)")
for _ in range(n_times):    
    x_0 = np.array([0, 1])
    start = time.time()
    x_opt_f2, u_opt_f2 = F_2(x_0, x_opt_f1, u_opt_f1)
    execution_time = time.time() - start
    execution_times_part3.append(execution_time)
    x_opt_f2 = x_opt_f2.full()
    u_opt_f2 = u_opt_f2.full()

# transform Opti stack to function (with initial guess provided for dual and primal variable)
p = opti.parameter(2)
F_3 = opti.to_function('F_3', [p, Xs, Us, opti.lam_g], [Xs, Us, opti.lam_g], ['initial_state', 'guess_state_traj', 'guess_controls', 'guess_dual_vars'], ['x_opt', 'u_opt', 'dual_vars'])

# solve the problem
print("\n\n\n\n Function with initial guess (primal and dual variable)")
for _ in range(n_times):    
    x_0 = np.array([0, 1])
    start = time.time()
    x_opt_f3, u_opt_f3, _ = F_3(x_0, x_opt_f2, u_opt_f2, lam_g0)
    execution_time = time.time() - start
    execution_times_part4.append(execution_time)
    x_opt_f3 = x_opt_f3.full()
    u_opt_f3 = u_opt_f3.full()

assert(x_opt.all()==x_opt_f1.all())
assert(u_opt.all()==u_opt_f1.all())
assert(x_opt.all()==x_opt_f2.all())
assert(u_opt.all()==u_opt_f2.all())
assert(x_opt.all()==x_opt_f3.all())
assert(u_opt.all()==u_opt_f3.all())


# Create plots
plt.figure(figsize=(12, 6))

# Plot average execution times
plt.subplot(1, 2, 1)
methods = ['Opti', 'Function1', 'Function2', 'Function3']
average_times = [sum(execution_times_part1) / n_times, sum(execution_times_part2) / n_times, sum(execution_times_part3) / n_times, sum(execution_times_part4) / n_times]
plt.bar(methods, average_times, color=['b', 'g', 'r', 'c'])
plt.ylabel('Average Execution Time (s)')
plt.title('Average Execution Times')

# Plot individual execution times
plt.subplot(1, 2, 2)
x = range(1, n_times + 1)
plt.plot(x, execution_times_part1, label='Opti', marker='o', color='b')
plt.plot(x, execution_times_part2, label='Function', marker='o', color='g')
plt.plot(x, execution_times_part3, label='Function (with initial guess prim)', marker='o', color='r')
plt.plot(x, execution_times_part4, label='Function (with initial guess prim and dual)', marker='o', color='c')
plt.xlabel('Execution Number')
plt.ylabel('Execution Time (s)')
plt.title('Execution Times for Each Execution')
plt.legend()

plt.tight_layout()
plt.show()
