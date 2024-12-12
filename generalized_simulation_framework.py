import numpy as np
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QWidget
import pyqtgraph as pg


def create_simulation_framework(
    step_simulation,
    transform_positions,
    default_state,
    dt,
    params,
    animation_range,
    input_labels,
    plot_labels,
):
    """
    Generalized simulation framework for particle systems.
    """
    t = 0.0
    running = False
    state = default_state
    t_data = []
    plot_data = {label: [] for label, _ in plot_labels}

    def initialize_simulation(inputs):
        nonlocal t, state, t_data, plot_data
        print("Initializing simulation...")

        # Parse inputs for initial conditions

        key1, key2 = inputs.keys()
        print(key1, key2)
        q1 = float(inputs[key1].text())
        q2 = float(inputs[key2].text())

        # Initialize state with zero angular velocities
        state = [q1, 0.0, q2, 0.0]  # Ensure 4 elements
        print(f"Initial state: {state}")

        # Reset time and plot data
        t = 0.0
        t_data.clear()
        for data in plot_data.values():
            data.clear()

        # Update animation with the new initial state
        positions = transform_positions(state, **params)
        update_plot(
            state,  # Pass state directly
            positions,
            gui["animation_items"],
            gui["plot_items"],
            t_data,
            plot_data,
        )
        print("Animation redrawn after initialization.")


    def update_plot(state, positions, animation_items, plot_items, t_data, plot_data, **kwargs):
        """
        Update the animation and plots with the current state.
        """
        x1, y1, x2, y2 = positions

        # Update animation
        animation_items["ball1"].setData([x1], [y1])
        animation_items["ball2"].setData([x2], [y2])
        animation_items["line1"].setData([0, x1], [0, y1])
        animation_items["line2"].setData([x1, x2], [y1, y2])

        # Update time-series plot
        t_data.append(len(t_data) * kwargs.get("dt", 0.02))

        data_keys = plot_data.keys()
        q1, q2 = data_keys

        plot_data[q1].append(state[0])
        plot_data[q2].append(state[2])
        plot_items[q1].setData(t_data, plot_data[q1])
        plot_items[q2].setData(t_data, plot_data[q2])

    def start_simulation(timer):
        print("Starting simulation...")
        nonlocal running
        if not running:
            running = True
            timer.start(int(dt * 1000))

    def stop_simulation(timer):
        nonlocal running
        if running:
            running = False
            timer.stop()

    def run_simulation(timer, animation_items, plot_items):
        nonlocal t, state
        if running:
            state = step_simulation(state, t, **params)
            print(f"Running simulation step at t = {t}, state = {state}")
            t += dt

            positions = transform_positions(state, **params)
            update_plot(
                state,  # Pass state directly
                positions,
                animation_items,
                plot_items,
                t_data,
                plot_data,
            )



    gui = create_gui(input_labels, plot_labels, animation_range)

    return {
        "gui": gui,
        "initialize": initialize_simulation,
        "start": start_simulation,
        "stop": stop_simulation,
        "run": run_simulation,
        "state": lambda: state,
    }


def create_gui(input_labels, plot_labels, animation_range):
    """
    Creates a reusable GUI framework for parameterized input, animation, and plot panels.
    """
    main_layout = QVBoxLayout()

    # Input Panel
    input_layout = QFormLayout()
    inputs = {label: QLineEdit("0.0") for label in input_labels}
    for label, widget in inputs.items():
        input_layout.addRow(label, widget)

    # Buttons
    button_layout = QHBoxLayout()
    initialize_button = QPushButton("Initialize")
    start_button = QPushButton("Start")
    stop_button = QPushButton("Stop")
    button_layout.addWidget(initialize_button)
    button_layout.addWidget(start_button)
    button_layout.addWidget(stop_button)

    # Animation Panel
    animation_widget = pg.PlotWidget()
    animation_widget.setBackground("#1e2127")
    animation_widget.setAspectLocked(True)
    animation_widget.setRange(xRange=animation_range, yRange=animation_range)
    animation_items = {
        "ball1": pg.ScatterPlotItem(size=20, brush="r"),
        "ball2": pg.ScatterPlotItem(size=20, brush="b"),
        "line1": pg.PlotDataItem(),
        "line2": pg.PlotDataItem(),
    }
    for item in animation_items.values():
        animation_widget.addItem(item)

    # Plot Panel
    plot_widget = pg.PlotWidget()
    plot_widget.setBackground("#1e2127")
    plot_widget.getAxis("bottom").setPen(pg.mkPen("#abb2bf"))
    plot_widget.getAxis("left").setPen(pg.mkPen("#abb2bf"))
    plot_items = {
        label: plot_widget.plot(pen=pg.mkPen(color=color, width=2))
        for label, color in plot_labels
    }

    # Combine layouts
    main_layout.addLayout(input_layout)
    main_layout.addLayout(button_layout)
    main_layout.addWidget(animation_widget)
    main_layout.addWidget(plot_widget)

    return {
        "layout": main_layout,
        "inputs": inputs,
        "buttons": {
            "initialize": initialize_button,
            "start": start_button,
            "stop": stop_button,
        },
        "animation_items": animation_items,
        "plot_items": plot_items,
    }