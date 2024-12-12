import sys
import numpy as np
from scipy.integrate import solve_ivp
from generalized_simulation_framework import create_simulation_framework

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QWidget
from PyQt5.QtCore import QTimer


def sliding_pendulum_ode(t, state, m, M, l, g):
    """
    Defines the equations of motion for the sliding pendulum.
    """
    x, x_dot, theta, theta_dot = state

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = M + m * sin_theta**2

    # Equations of motion
    x_ddot = (
        -m * l * theta_dot**2 * sin_theta + m * g * sin_theta * cos_theta
    ) / denominator
    theta_ddot = (
        -g * sin_theta - cos_theta * x_ddot
    ) / l

    return [x_dot, x_ddot, theta_dot, theta_ddot]


def sliding_pendulum_positions(state, **kwargs):
    """
    Map the generalized coordinates to Cartesian positions.
    """
    l = kwargs["l"]  # Length of the pendulum
    x, _, theta, _ = state

    # Cart position
    cart_x = x
    cart_y = 0.0

    # Pendulum bob position
    pendulum_x = x + l * np.sin(theta)
    pendulum_y = -l * np.cos(theta)

    return [cart_x, cart_y, pendulum_x, pendulum_y]





def create_sliding_pendulum_simulation():
    m, M, l, g = 1.0, 5.0, 1.0, 9.81
    dt = 0.02
    default_state = [0.0, 0.0, 0.1, 0.0]  # Initial cart position, velocity, angle, angular velocity

    # Define step simulation function
    def step_simulation(state, t, m, M, l, g):
        t_span = [t, t + dt]
        sol = solve_ivp(
            fun=lambda t, y: sliding_pendulum_ode(t, y, m, M, l, g),
            t_span=t_span,
            y0=state,
            t_eval=[t + dt],
        )
        return sol.y[:, -1]

    return create_simulation_framework(
        step_simulation=step_simulation,
        transform_positions=sliding_pendulum_positions,
        default_state=default_state,
        dt=dt,
        params={"m": m, "M": M, "l": l, "g": g},
        animation_range=(-3, 3),
        input_labels=["Initial x (m)", "Initial θ (rad)"],
        plot_labels=[("x(t)", "#56b6c2"), ("θ(t)", "#e06c75")],
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create sliding pendulum simulation
    simulation = create_sliding_pendulum_simulation()
    gui = simulation["gui"]

    # Timer for simulation updates
    timer = QTimer()

    # Connect buttons to their respective actions
    gui["buttons"]["initialize"].clicked.connect(
        lambda: (print("Initialize clicked"), simulation["initialize"](gui["inputs"]))
    )
    gui["buttons"]["start"].clicked.connect(
        lambda: (print("Start clicked"), simulation["start"](timer))
    )
    gui["buttons"]["stop"].clicked.connect(
        lambda: (print("Stop clicked"), simulation["stop"](timer))
    )

    # Connect the timer to the simulation runner
    timer.timeout.connect(
        lambda: simulation["run"](timer, gui["animation_items"], gui["plot_items"])
    )

    # Create and display the main window
    window = QMainWindow()
    container = QWidget()
    container.setLayout(gui["layout"])
    window.setCentralWidget(container)
    window.show()

    print("GUI is ready. Waiting for user interaction...")
    sys.exit(app.exec_())
