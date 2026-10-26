"""
DOC-Oc Precision Continuum Robot Controller

CONTROLS
--------

        W
        ↑
        |
A  ←---+---→  D
        |
        ↓
        S

W : +Y
S : -Y
A : -X
D : +X

IMPORTANT
---------

The four tendons are controlled antagonistically.

D:
    +X tendon shortens
    -X tendon releases

A:
    +X tendon releases
    -X tendon shortens

W:
    +Y tendon shortens
    -Y tendon releases

S:
    +Y tendon releases
    -Y tendon shortens

Therefore:

    D D D A A A

returns exactly to the neutral configuration.

The same applies to W/S.

One keyboard event = exactly one motor step = 0.18 deg.
"""

from __future__ import annotations

import math
import queue
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from .measurement import (
    get_measurements,
    get_nitinol_tendon_lengths,
)


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "scene.xml"
)

PARAM_PATH = (
    PROJECT_ROOT
    / "parameters"
    / "robot_params.yaml"
)


# =====================================================================
# CONSTANTS
# =====================================================================

MOTOR_NAMES = (
    "px",
    "nx",
    "py",
    "ny",
)


# Keyboard -> signed Cartesian axis
#
# D = +X
# A = -X
# W = +Y
# S = -Y

KEY_COMMANDS = {
    "d": ("x", +1),
    "a": ("x", -1),
    "w": ("y", +1),
    "s": ("y", -1),
}


# =====================================================================
# KEYBOARD QUEUE
# =====================================================================

COMMAND_QUEUE = queue.Queue()


def key_callback(keycode):

    """
    MuJoCo passive viewer keyboard callback.

    The official MuJoCo Python API passes the keyboard keycode here.
    """

    try:

        key = chr(
            int(keycode)
        ).lower()

    except (ValueError, TypeError):

        return

    if key in KEY_COMMANDS:

        COMMAND_QUEUE.put(
            key
        )


# =====================================================================
# PARAMETERS
# =====================================================================

def load_parameters():

    with open(
        PARAM_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


# =====================================================================
# MUJOCO IDS
# =====================================================================

def actuator_id(
    model,
    name,
):

    return model.actuator(
        f"motor_{name}"
    ).id


def tendon_id(
    model,
    name,
):

    return model.tendon(
        f"tendon_{name}"
    ).id


# =====================================================================
# INITIAL TENDON LENGTHS
# =====================================================================

def get_initial_tendon_lengths(
    model,
    data,
):

    lengths = {}

    for name in MOTOR_NAMES:

        tid = tendon_id(
            model,
            name,
        )

        lengths[name] = float(
            data.ten_length[tid]
        )

    return lengths


# =====================================================================
# INITIAL MOTOR TARGETS
# =====================================================================

def initialize_targets(
    model,
    data,
):

    targets = (
        get_initial_tendon_lengths(
            model,
            data,
        )
    )

    for name in MOTOR_NAMES:

        aid = actuator_id(
            model,
            name,
        )

        data.ctrl[aid] = (
            targets[name]
        )

    return targets


# =====================================================================
# ANTIGONISTIC TENDON CONTROL
# =====================================================================

def update_axis_targets(
    model,
    data,
    neutral_lengths,
    x_steps,
    y_steps,
    step_displacement_m,
):
    """
    Convert signed Cartesian commands into four tendon targets.

    This is the key change.

    x_steps > 0:

        +X tendon shortens
        -X tendon lengthens

    x_steps < 0:

        +X tendon lengthens
        -X tendon shortens

    Same logic for Y.
    """

    x_delta = (
        x_steps
        *
        step_displacement_m
    )

    y_delta = (
        y_steps
        *
        step_displacement_m
    )

    targets = {

        # +X
        "px":
            neutral_lengths["px"]
            -
            x_delta,

        # -X
        "nx":
            neutral_lengths["nx"]
            +
            x_delta,

        # +Y
        "py":
            neutral_lengths["py"]
            -
            y_delta,

        # -Y
        "ny":
            neutral_lengths["ny"]
            +
            y_delta,
    }

    for name in MOTOR_NAMES:

        aid = actuator_id(
            model,
            name,
        )

        data.ctrl[aid] = (
            targets[name]
        )

    return targets


# =====================================================================
# ONE COMMAND
# =====================================================================

def execute_command(
    model,
    data,
    viewer,
    neutral_lengths,
    x_steps,
    y_steps,
    command,
    step_displacement_m,
    settle_steps,
):
    """
    Apply exactly one directional command.
    """

    axis, direction = (
        KEY_COMMANDS[
            command
        ]
    )

    if axis == "x":

        x_steps += direction

    elif axis == "y":

        y_steps += direction


    # ---------------------------------------------------------------
    # Apply new antagonistic tendon state
    # ---------------------------------------------------------------

    targets = update_axis_targets(
        model,
        data,
        neutral_lengths,
        x_steps,
        y_steps,
        step_displacement_m,
    )


    # ---------------------------------------------------------------
    # Run physics ONLY after this keyboard event
    # ---------------------------------------------------------------

    for i in range(
        settle_steps
    ):

        mujoco.mj_step(
            model,
            data,
        )

        if i % 20 == 0:

            viewer.sync()


    return (
        x_steps,
        y_steps,
        targets,
    )


# =====================================================================
# NITINOL SAFETY
# =====================================================================

def get_nitinol_status(
    strain,
):

    if strain >= 0.08:

        return (
            "CRITICAL — 8% LIMIT",
            "critical",
        )

    if strain >= 0.07:

        return (
            "WARNING — HIGH STRAIN",
            "warning",
        )

    return (
        "SAFE",
        "safe",
    )


# =====================================================================
# TERMINAL DISPLAY
# =====================================================================

def print_state(
    measurements,
    x_steps,
    y_steps,
    command,
):

    tip = measurements[
        "tip_position_m"
    ]

    pcc = measurements[
        "pcc"
    ]

    max_strain = measurements[
        "max_nitinol_tensile_strain"
    ]

    status, _ = get_nitinol_status(
        max_strain
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"COMMAND       : "
        f"{command.upper()}"
    )

    print(
        f"AXIS STATE    : "
        f"X={x_steps:+d} "
        f"Y={y_steps:+d}"
    )

    print(
        f"TIP X         : "
        f"{tip[0] * 1000:.6f} mm"
    )

    print(
        f"TIP Y         : "
        f"{tip[1] * 1000:.6f} mm"
    )

    print(
        f"TIP Z         : "
        f"{tip[2] * 1000:.6f} mm"
    )

    print(
        f"TIP XY        : "
        f"{measurements['tip_distance_xy_m'] * 1000:.6f} mm"
    )

    print(
        f"BEND ANGLE    : "
        f"{pcc.bending_angle_deg:.6f} deg"
    )

    print(
        f"CURVATURE     : "
        f"{pcc.curvature_1_m:.8f} 1/m"
    )

    print(
        f"CURVATURE X   : "
        f"{pcc.curvature_x_1_m:.8f} 1/m"
    )

    print(
        f"CURVATURE Y   : "
        f"{pcc.curvature_y_1_m:.8f} 1/m"
    )

    print(
        f"DIRECTION     : "
        f"{pcc.direction_deg:.6f} deg"
    )

    print(
        f"NITINOL MAX   : "
        f"{max_strain * 100:.6f} %"
    )

    print(
        f"STATUS        : "
        f"{status}"
    )

    print(
        "=" * 80
    )


# =====================================================================
# VIEWER OVERLAY
# =====================================================================

def update_overlay(
    viewer,
    measurements,
    x_steps,
    y_steps,
    last_command,
):
    """
    Display precision robot information directly in the
    MuJoCo passive viewer.

    MuJoCo viewer.set_texts() expects:

        (font, gridpos, text1, text2)
    """

    tip = measurements["tip_position_m"]

    pcc = measurements["pcc"]

    max_strain = (
        measurements[
            "max_nitinol_tensile_strain"
        ]
    )

    status, _ = get_nitinol_status(
        max_strain
    )

    wire_strains = (
        np.maximum(
            measurements[
                "nitinol_strains"
            ],
            0.0,
        )
        * 100.0
    )

    last = (
        last_command.upper()
        if last_command
        else "NONE"
    )

    # --------------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------------

    text_left = f"""
DOC-Oc PRECISION CONTROLLER

CONTROLS
W  = +Y
S  = -Y
A  = -X
D  = +X

LAST COMMAND
{last}

AXIS STATE
X : {x_steps:+d} steps
Y : {y_steps:+d} steps

TIP POSITION
X : {tip[0] * 1000.0:9.5f} mm
Y : {tip[1] * 1000.0:9.5f} mm
Z : {tip[2] * 1000.0:9.5f} mm

TIP XY DISTANCE
{measurements["tip_distance_xy_m"] * 1000.0:9.5f} mm
"""

    # --------------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------------

    text_right = f"""
PCC KINEMATICS

BENDING ANGLE
{pcc.bending_angle_deg:9.6f} deg

CURVATURE
{pcc.curvature_1_m:9.8f} 1/m

CURVATURE X
{pcc.curvature_x_1_m:9.8f} 1/m

CURVATURE Y
{pcc.curvature_y_1_m:9.8f} 1/m

BENDING DIRECTION
{pcc.direction_deg:9.6f} deg


NITINOL STRAIN

W1 : {wire_strains[0]:8.5f} %
W2 : {wire_strains[1]:8.5f} %
W3 : {wire_strains[2]:8.5f} %
W4 : {wire_strains[3]:8.5f} %
W5 : {wire_strains[4]:8.5f} %
W6 : {wire_strains[5]:8.5f} %
W7 : {wire_strains[6]:8.5f} %

MAX STRAIN
{max_strain * 100.0:9.6f} %

STATUS
{status}
"""

    # --------------------------------------------------------------
    # MuJoCo viewer API
    # --------------------------------------------------------------

    viewer.set_texts(
        [
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                text_left,
                text_right,
            )
        ]
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "DOC-Oc PRECISION CONTINUUM CONTROLLER"
    )

    print(
        "=" * 80
    )


    # ---------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------

    params = load_parameters()

    geometry = params[
        "geometry"
    ]

    motors = params[
        "motors"
    ]

    nitinol = params[
        "nitinol"
    ]

    control = params.get(
        "control",
        {},
    )


    n_disks = int(
        geometry.get(
            "number_of_discs",
            11,
        )
    )

    tendon_radius_m = float(
        geometry[
            "tendon_radius_m"
        ]
    )

    spool_radius_m = float(
        motors[
            "spool_radius_m"
        ]
    )

    step_angle_deg = float(
        motors.get(
            "step_angle_deg",
            0.18,
        )
    )

    warning_strain = float(
        nitinol.get(
            "warning_strain",
            0.07,
        )
    )

    damage_limit = float(
        nitinol.get(
            "damage_limit_strain",
            0.08,
        )
    )

    settle_steps = int(
        control.get(
            "settle_steps",
            1500,
        )
    )


    # ---------------------------------------------------------------
    # One motor step
    # ---------------------------------------------------------------

    step_displacement_m = (
        spool_radius_m
        *
        math.radians(
            step_angle_deg
        )
    )


    print(
        f"Motor step             : "
        f"{step_angle_deg:.4f} deg"
    )

    print(
        f"Spool radius           : "
        f"{spool_radius_m*1000:.4f} mm"
    )

    print(
        f"Tendon movement/step   : "
        f"{step_displacement_m*1e6:.6f} um"
    )

    print(
        f"Warning strain         : "
        f"{warning_strain*100:.2f}%"
    )

    print(
        f"Hard strain limit      : "
        f"{damage_limit*100:.2f}%"
    )


    # ---------------------------------------------------------------
    # Load MuJoCo
    # ---------------------------------------------------------------

    print()
    print(
        "Loading:"
    )

    print(
        MODEL_PATH
    )

    model = (
        mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )
    )

    data = mujoco.MjData(
        model
    )


    # ---------------------------------------------------------------
    # Initial forward pass
    # ---------------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )


    # ---------------------------------------------------------------
    # Neutral tendon lengths
    # ---------------------------------------------------------------

    neutral_lengths = (
        initialize_targets(
            model,
            data,
        )
    )


    print()
    print(
        "NEUTRAL TENDON LENGTHS"
    )

    for name in MOTOR_NAMES:

        print(
            f"{name:>3} : "
            f"{neutral_lengths[name]*1000:.6f} mm"
        )


    # ---------------------------------------------------------------
    # Nitinol rest lengths
    # ---------------------------------------------------------------

    nitinol_rest_lengths = (
        get_nitinol_tendon_lengths(
            model,
            data,
            count=7,
        )
    )


    print()
    print(
        "NITINOL REFERENCE LENGTHS"
    )

    for i, length in enumerate(
        nitinol_rest_lengths,
        start=1,
    ):

        print(
            f"W{i} : "
            f"{length*1000:.6f} mm"
        )


    # ---------------------------------------------------------------
    # Axis state
    # ---------------------------------------------------------------

    x_steps = 0
    y_steps = 0

    last_command = None


    # ---------------------------------------------------------------
    # Viewer
    # ---------------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:


        # -----------------------------------------------------------
        # Camera
        # -----------------------------------------------------------

        viewer.cam.type = (
            mujoco.mjtCamera.mjCAMERA_FREE
        )

        viewer.cam.lookat[:] = [
            0.0,
            0.0,
            0.050,
        ]

        viewer.cam.distance = 0.23

        viewer.cam.azimuth = 135

        viewer.cam.elevation = -10


        # -----------------------------------------------------------
        # Initial measurements
        # -----------------------------------------------------------

        measurements = get_measurements(
            model,
            data,
            tendon_radius_m,
            n_disks,
            nitinol_rest_lengths,
        )


        update_overlay(
            viewer,
            measurements,
            x_steps,
            y_steps,
            None,
        )

        viewer.sync()


        # ===========================================================
        # EVENT LOOP
        # ===========================================================

        while viewer.is_running():

            command = None

            try:

                command = (
                    COMMAND_QUEUE.get_nowait()
                )

            except queue.Empty:

                command = None


            # -------------------------------------------------------
            # NO KEY:
            #
            # DO NOT STEP PHYSICS.
            # -------------------------------------------------------

            if command is None:

                time.sleep(
                    0.005
                )

                continue


            # -------------------------------------------------------
            # Check Nitinol before command
            # -------------------------------------------------------

            measurements = get_measurements(
                model,
                data,
                tendon_radius_m,
                n_disks,
                nitinol_rest_lengths,
            )

            current_strain = (
                measurements[
                    "max_nitinol_tensile_strain"
                ]
            )


            if (
                current_strain
                >=
                damage_limit
            ):

                print()
                print(
                    "================================================"
                )

                print(
                    "NITINOL PROTECTION ACTIVE"
                )

                print(
                    f"Current strain: "
                    f"{current_strain*100:.5f}%"
                )

                print(
                    "8% limit reached."
                )

                print(
                    "W/A/S/D commands are blocked."
                )

                print(
                    "================================================"
                )

                continue


            # -------------------------------------------------------
            # Execute exactly ONE directional step
            # -------------------------------------------------------

            with viewer.lock():

                (
                    x_steps,
                    y_steps,
                    targets,
                ) = execute_command(
                    model,
                    data,
                    viewer,
                    neutral_lengths,
                    x_steps,
                    y_steps,
                    command,
                    step_displacement_m,
                    settle_steps,
                )


            last_command = command


            # -------------------------------------------------------
            # Measure actual MuJoCo state
            # -------------------------------------------------------

            measurements = get_measurements(
                model,
                data,
                tendon_radius_m,
                n_disks,
                nitinol_rest_lengths,
            )


            # -------------------------------------------------------
            # Nitinol status
            # -------------------------------------------------------

            current_strain = (
                measurements[
                    "max_nitinol_tensile_strain"
                ]
            )

            if (
                current_strain
                >=
                damage_limit
            ):

                print()
                print(
                    "!!! CRITICAL NITINOL STRAIN !!!"
                )

                print(
                    f"{current_strain*100:.5f}%"
                )

            elif (
                current_strain
                >=
                warning_strain
            ):

                print()
                print(
                    "!!! WARNING: NITINOL STRAIN HIGH !!!"
                )

                print(
                    f"{current_strain*100:.5f}%"
                )


            # -------------------------------------------------------
            # Print
            # -------------------------------------------------------

            print_state(
                measurements,
                x_steps,
                y_steps,
                command,
            )


            # -------------------------------------------------------
            # Update GUI
            # -------------------------------------------------------

            update_overlay(
                viewer,
                measurements,
                x_steps,
                y_steps,
                last_command,
            )

            viewer.sync()


            time.sleep(
                0.01
            )


if __name__ == "__main__":

    main()