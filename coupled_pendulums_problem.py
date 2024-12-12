import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Step 1: Define symbols and parameters
t = sp.Symbol("t")
theta1 = sp.Function("theta1")(t)
theta2 = sp.Function("theta2")(t)
theta1_dot = theta1.diff(t)
theta2_dot = theta2.diff(t)

# Define second derivatives as unknowns
theta1_ddot = sp.symbols("theta1_ddot")
theta2_ddot = sp.symbols("theta2_ddot")

# Parameters
g = sp.symbols("g")
m1, m2 = sp.symbols("m1 m2")
l1, l2 = sp.symbols("l1 l2")

# Coordinates of the pendulum masses
x1 = l1 * sp.sin(theta1)
y1 = -l1 * sp.cos(theta1)
x2 = l1 * sp.sin(theta1) + l2 * sp.sin(theta2)
y2 = -l1 * sp.cos(theta1) - l2 * sp.cos(theta2)

# Velocities
vx1 = x1.diff(t)
vy1 = y1.diff(t)
vx2 = x2.diff(t)
vy2 = y2.diff(t)

# Kinetic energy
T1 = (1 / 2) * m1 * (vx1**2 + vy1**2)
T2 = (1 / 2) * m2 * (vx2**2 + vy2**2)
T = sp.simplify(T1 + T2)

# Potential energy
V1 = m1 * g * (-y1)
V2 = m2 * g * (-y2)
V = sp.simplify(V1 + V2)

# Lagrangian
L = T - V

# Euler-Lagrange equations
eom1 = sp.Eq(sp.diff(L.diff(theta1_dot), t) - L.diff(theta1), 0)
eom2 = sp.Eq(sp.diff(L.diff(theta2_dot), t) - L.diff(theta2), 0)

# Substitute second derivatives as symbols
eom1 = eom1.subs({theta1.diff(t, 2): theta1_ddot, theta2.diff(t, 2): theta2_ddot})
eom2 = eom2.subs({theta1.diff(t, 2): theta1_ddot, theta2.diff(t, 2): theta2_ddot})

# Solve for the second derivatives
sol = sp.solve([eom1, eom2], [theta1_ddot, theta2_ddot])
theta1_ddot_expr = sp.simplify(sol[theta1_ddot])
theta2_ddot_expr = sp.simplify(sol[theta2_ddot])

# Step 2: Precompute lambdified functions for numerical evaluation
theta1_ddot_func = sp.lambdify(
    (theta1, theta1_dot, theta2, theta2_dot, l1, l2, m1, m2, g),
    theta1_ddot_expr,
    modules="numpy",
)
theta2_ddot_func = sp.lambdify(
    (theta1, theta1_dot, theta2, theta2_dot, l1, l2, m1, m2, g),
    theta2_ddot_expr,
    modules="numpy",
)

# Step 3: Define the ODE system
def odes(t, y, l1, l2, m1, m2, g):
    theta1, theta1_dot, theta2, theta2_dot = y
    theta1_ddot = theta1_ddot_func(theta1, theta1_dot, theta2, theta2_dot, l1, l2, m1, m2, g)
    theta2_ddot = theta2_ddot_func(theta1, theta1_dot, theta2, theta2_dot, l1, l2, m1, m2, g)
    return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

# Step 4: Solve the system numerically with solve_ivp
y0 = [np.pi / 4, 0, np.pi / 6, 0]  # Initial conditions: [theta1, theta1_dot, theta2, theta2_dot]
t_span = (0, 10)  # Time range in seconds
t_eval = np.linspace(*t_span, 1000)  # Time points to evaluate the solution

# Fixed parameters for the numerical system
params = {
    "l1": 1.0,
    "l2": 1.0,
    "m1": 1.0,
    "m2": 1.0,
    "g": 9.81
}

# Use solve_ivp to solve the system
sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, args=(params["l1"], params["l2"], params["m1"], params["m2"], params["g"]))

# Step 5: Plot the results
plt.plot(sol.t, sol.y[0], label=r'$\theta_1$ (rad)')
plt.plot(sol.t, sol.y[2], label=r'$\theta_2$ (rad)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Double Pendulum Dynamics')
plt.legend()
plt.grid()
plt.show()
