"""DOC-Oc MuJoCo controller.

Keyboard behaviour
------------------
W = +Y bending
S = -Y bending
A = -X bending
D = +X bending

A physical key-down produces exactly one 0.18 degree motor step.
Keeping the key physically held produces additional discrete steps
at a fixed repeat period. Releasing the key stops further commands
immediately.

The keyboard listener suppresses W/A/S/D from reaching MuJoCo's own
viewer shortcuts. This is important because the MuJoCo viewer uses
keyboard shortcuts of its own (including rendering/camera controls).

Run on macOS with:
    mjpython -m src.main
"""

from pathlib import Path
import math
import time
from threading import Lock

import numpy as np
import yaml
import mujoco
import mujoco.viewer
from pynput import keyboard

from .kinematics import (
    motor_step_displacement,
    solve_pcc_from_tendon_lengths,
)
from .measurement import get_measurements


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "scene.xml"
PARAM_PATH = PROJECT_ROOT / "parameters" / "robot_params.yaml"

COMMANDS = {"w", "a", "s", "d"}


# ============================================================
# KEYBOARD STATE
# ============================================================

class KeyboardState:
    """Thread-safe physical WASD state.

    pynput callbacks run on a separate OS listener thread.
    The simulation thread consumes pending first-press events and
    repeatedly checks the currently held keys.
    """

    def __init__(self):
        self._lock = Lock()
        self._held = set()
        self._pending_press = set()
        self._last_repeat = {}
        self._last_event = time.perf_counter()

    def press(self, key):
        if key not in COMMANDS:
            return

        now = time.perf_counter()

        with self._lock:
            # Ignore OS auto-repeat. Only the physical transition
            # from UP -> DOWN is accepted as a new press.
            if key in self._held:
                return

            self._held.add(key)
            self._pending_press.add(key)
            self._last_repeat[key] = now
            self._last_event = now

    def release(self, key):
        if key not in COMMANDS:
            return

        with self._lock:
            self._held.discard(key)
            self._pending_press.discard(key)
            self._last_repeat.pop(key, None)
            self._last_event = time.perf_counter()

    def consume_new_presses(self):
        with self._lock:
            result = tuple(self._pending_press)
            self._pending_press.clear()
            return result

    def due_repeats(self, period):
        now = time.perf_counter()
        due = []

        with self._lock:
            for key in tuple(self._held):
                last = self._last_repeat.get(key, now)

                if now - last >= period:
                    self._last_repeat[key] = now
                    due.append(key)

        return due

    def active_text(self):
        with self._lock:
            return "".join(sorted(self._held)).upper() or "NONE"

    def clear(self):
        with self._lock:
            self._held.clear()
            self._pending_press.clear()
            self._last_repeat.clear()


# ============================================================
# ARM CONTROLLER
# ============================================================

class Controller:
    def __init__(self, model, data, params):
        self.model = model
        self.data = data

        geometry = params["geometry"]
        motor = params["motor"]
        control = params["control"]
        nitinol = params["nitinol"]

        self.L = float(geometry["arm_centerline_length_m"])
        self.r = float(geometry["tendon_radius_m"])

        self.repeat_period = float(
            control["key_repeat_period_s"]
        )

        self.max_x = int(
            control["max_command_steps_x"]
        )
        self.max_y = int(
            control["max_command_steps_y"]
        )

        self.warning_strain = float(
            nitinol["warning_strain"]
        )
        self.hard_strain_limit = float(
            nitinol["hard_strain_limit"]
        )

        # Nitinol undeformed/free length. This is deliberately NOT
        # the already-pretensioned length measured at startup.
        # The XML springlength is the free/reference length used for
        # the first-order elastic tendon representation.
        self.nitinol_free_length = float(
            nitinol.get(
                "free_length_m",
                0.0959804,
            )
        )

        self.step_length = motor_step_displacement(
            float(motor["spool_radius_m"]),
            float(motor["step_angle_deg"]),
        )

        self.x_steps = 0
        self.y_steps = 0
        self.total_steps = 0
        self.last_command = "NONE"
        self.safety_latched = False

        # --------------------------------------------------------
        # Validate model names first.
        # --------------------------------------------------------

        for name in ("px", "nx", "py", "ny"):
            tendon_name = f"tendon_{name}"
            actuator_name = f"motor_{name}"

            if model.tendon(tendon_name).id < 0:
                raise RuntimeError(
                    f"Missing tendon in scene.xml: {tendon_name}"
                )

            if model.actuator(actuator_name).id < 0:
                raise RuntimeError(
                    f"Missing actuator in scene.xml: {actuator_name}"
                )

        # --------------------------------------------------------
        # Read actual neutral tendon lengths from the compiled
        # model. These include the exact current geometry.
        # --------------------------------------------------------

        mujoco.mj_forward(model, data)

        self.neutral = {
            name: float(
                data.ten_length[
                    model.tendon(f"tendon_{name}").id
                ]
            )
            for name in ("px", "nx", "py", "ny")
        }

        self.targets = dict(self.neutral)

        # --------------------------------------------------------
        # CRITICAL: actuator controls must be initialized to the
        # actual neutral tendon lengths before the first mj_step.
        # Otherwise a position actuator can start at ctrl=0 and
        # immediately pull the robot into a large bend.
        # --------------------------------------------------------

        self.write_targets()
        mujoco.mj_forward(model, data)

    # ========================================================
    # WRITE ACTUATOR TARGETS
    # ========================================================

    def write_targets(self):
        for name in ("px", "nx", "py", "ny"):
            aid = self.model.actuator(f"motor_{name}").id

            lo = float(
                self.model.actuator_ctrlrange[aid, 0]
            )
            hi = float(
                self.model.actuator_ctrlrange[aid, 1]
            )

            self.targets[name] = float(
                np.clip(
                    self.targets[name],
                    lo,
                    hi,
                )
            )

            self.data.ctrl[aid] = self.targets[name]

    # ========================================================
    # NITINOL STRAIN
    # ========================================================

    def nitinol_strains(self):
        """Return engineering strain relative to free wire length."""

        free = self.nitinol_free_length

        if free <= 0.0:
            raise ValueError(
                "Nitinol free length must be positive."
            )

        lengths = np.array([
            float(
                self.data.ten_length[
                    self.model.tendon(
                        f"central_wire_{i}"
                    ).id
                ]
            )
            for i in range(1, 8)
        ], dtype=float)

        return (lengths - free) / free

    # ========================================================
    # ONE EXACT MOTOR STEP
    # ========================================================

    def execute_step(self, command):
        if self.safety_latched:
            return

        d = 0.5 * self.step_length

        # ----------------------------------------------------
        # D = +X
        # ----------------------------------------------------

        if command == "d":
            if self.x_steps >= self.max_x:
                return

            self.targets["px"] -= d
            self.targets["nx"] += d
            self.x_steps += 1

        # ----------------------------------------------------
        # A = -X
        # ----------------------------------------------------

        elif command == "a":
            if self.x_steps <= -self.max_x:
                return

            self.targets["px"] += d
            self.targets["nx"] -= d
            self.x_steps -= 1

        # ----------------------------------------------------
        # W = +Y
        # ----------------------------------------------------

        elif command == "w":
            if self.y_steps >= self.max_y:
                return

            self.targets["py"] -= d
            self.targets["ny"] += d
            self.y_steps += 1

        # ----------------------------------------------------
        # S = -Y
        # ----------------------------------------------------

        elif command == "s":
            if self.y_steps <= -self.max_y:
                return

            self.targets["py"] += d
            self.targets["ny"] -= d
            self.y_steps -= 1

        else:
            return

        self.write_targets()

        self.total_steps += 1
        self.last_command = command.upper()

    # ========================================================
    # SAFETY
    # ========================================================

    def trip_safety(self):
        if self.safety_latched:
            return

        self.safety_latched = True
        self.last_command = "SAFETY LOCK"

        print(
            "NITINOL HARD LIMIT REACHED: motion commands locked."
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset_command(self):
        self.targets = dict(self.neutral)
        self.x_steps = 0
        self.y_steps = 0
        self.total_steps = 0
        self.last_command = "RESET"
        self.safety_latched = False
        self.write_targets()
        mujoco.mj_forward(self.model, self.data)


# ============================================================
# CAMERA
# ============================================================

def setup_camera(viewer):
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.050]
        viewer.cam.distance = 0.145
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -10.0


# ============================================================
# OVERLAY
# ============================================================

def build_overlay(
    controller,
    keyboard_state,
    measurements,
    pcc,
):
    p = measurements["tip_position_m"]

    # Engineering strain relative to the free/reference Nitinol
    # length, not relative to the already-pretensioned startup state.
    strain = measurements["nitinol_strains"] * 100.0
    max_strain = (
        measurements["max_nitinol_tensile_strain"] * 100.0
    )

    if controller.safety_latched:
        status = "HARD LIMIT / LOCKED"
    elif max_strain >= controller.warning_strain * 100.0:
        status = "WARNING"
    else:
        status = "OK"

    return (
        "DOC-OC CONTINUUM ARM\n"
        "\n"
        f"ACTIVE KEYS : {keyboard_state.active_text()}\n"
        f"LAST COMMAND: {controller.last_command}\n"
        f"TOTAL STEPS : {controller.total_steps}\n"
        f"X STEPS     : {controller.x_steps:+d}\n"
        f"Y STEPS     : {controller.y_steps:+d}\n"
        "\n"
        f"TIP X       : {p[0]*1000:8.3f} mm\n"
        f"TIP Y       : {p[1]*1000:8.3f} mm\n"
        f"TIP Z       : {p[2]*1000:8.3f} mm\n"
        f"TIP XY      : {measurements['tip_distance_xy_m']*1000:8.3f} mm\n"
        "\n"
        f"CURVATURE   : {pcc.curvature_1_m:8.4f} 1/m\n"
        f"KAPPA X     : {pcc.curvature_x_1_m:8.4f} 1/m\n"
        f"KAPPA Y     : {pcc.curvature_y_1_m:8.4f} 1/m\n"
        f"BEND ANGLE  : {pcc.bending_angle_deg:8.3f} deg\n"
        f"DIRECTION   : {pcc.direction_deg:8.3f} deg\n"
        f"AXIAL STRAIN: {pcc.axial_strain*100:8.4f} %\n"
        "\n"
        f"W1 {strain[0]:7.3f}%   W2 {strain[1]:7.3f}%\n"
        f"W3 {strain[2]:7.3f}%   W4 {strain[3]:7.3f}%\n"
        f"W5 {strain[4]:7.3f}%   W6 {strain[5]:7.3f}%\n"
        f"W7 {strain[6]:7.3f}%\n"
        f"MAX NITINOL: {max_strain:7.3f}% [{status}]\n"
        "\n"
        "W = +Y     S = -Y\n"
        "A = -X     D = +X\n"
        "TAP = 1 STEP    HOLD = CONTINUOUS"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("DOC-OC CONTINUUM ARM")
    print("=" * 72)
    print(f"Model: {MODEL_PATH}")

    with open(PARAM_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )
    data = mujoco.MjData(model)

    controller = Controller(
        model,
        data,
        params,
    )

    # --------------------------------------------------------
    # Visual: make every CAD disc explicitly white.
    # The tendon/wire colors remain unchanged.
    # --------------------------------------------------------
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, gid
        ) or ""
        if gname.startswith("disc_geom_"):
            model.geom_rgba[gid] = np.array(
                [1.0, 1.0, 1.0, 1.0], dtype=float
            )

    keyboard_state = KeyboardState()

    # --------------------------------------------------------
    # pynput callbacks
    # --------------------------------------------------------

    def normalize_key(key):
        try:
            if isinstance(key, keyboard.KeyCode):
                if key.char:
                    return key.char.lower()
        except Exception:
            pass
        return None

    def on_press(key):
        k = normalize_key(key)
        if k in COMMANDS:
            keyboard_state.press(k)

    def on_release(key):
        k = normalize_key(key)
        if k in COMMANDS:
            keyboard_state.release(k)

    # suppress=True is intentional:
    # W/A/S/D must NOT simultaneously operate MuJoCo viewer
    # shortcuts such as wireframe/camera controls.
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
        suppress=True,
    )

    listener.daemon = True
    listener.start()

    print()
    print("CONTROLS")
    print("----------------------------------------")
    print("W = +Y")
    print("S = -Y")
    print("A = -X")
    print("D = +X")
    print()
    print("Tap  = exactly one 0.18 deg step")
    print("Hold = continuous discrete steps")
    print("Release = immediate stop")
    print()
    print(
        f"Step tendon travel = {controller.step_length*1e6:.3f} um"
    )
    print(
        f"Repeat period      = {controller.repeat_period*1000:.1f} ms"
    )
    print(
        f"Repeat rate        = {1.0/controller.repeat_period:.2f} steps/s"
    )
    print(
        f"Nitinol free length = {controller.nitinol_free_length*1000:.4f} mm"
    )
    print("----------------------------------------")
    print()

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:

            setup_camera(viewer)
            viewer.sync()

            next_viewer = time.perf_counter()
            next_overlay = time.perf_counter()

            # ------------------------------------------------
            # Main simulation loop
            # ------------------------------------------------

            while viewer.is_running():
                now = time.perf_counter()

                # =============================================
                # 1. EXACTLY ONE STEP FOR EACH NEW KEY-DOWN
                # =============================================

                new_presses = (
                    keyboard_state.consume_new_presses()
                )

                # =============================================
                # 2. CONTINUOUS STEPS WHILE KEYS ARE HELD
                # =============================================

                repeat_steps = keyboard_state.due_repeats(
                    controller.repeat_period
                )

                if new_presses or repeat_steps:
                    with viewer.lock():
                        for key in new_presses:
                            controller.execute_step(key)

                        for key in repeat_steps:
                            controller.execute_step(key)

                # =============================================
                # 3. ONE PHYSICS STEP
                # =============================================

                with viewer.lock():
                    mujoco.mj_step(
                        model,
                        data,
                    )

                    # =========================================
                    # 4. MEASUREMENTS
                    # =========================================

                    measurements = get_measurements(
                        model,
                        data,
                        controller.nitinol_free_length,
                    )

                    # =========================================
                    # 5. NITINOL HARD-LIMIT CHECK
                    # =========================================

                    if (
                        measurements[
                            "max_nitinol_tensile_strain"
                        ]
                        >= controller.hard_strain_limit
                    ):
                        controller.trip_safety()

                    # =========================================
                    # 6. PCC KINEMATICS
                    # =========================================

                    pcc = solve_pcc_from_tendon_lengths(
                        measurements["tendon_lengths_m"],
                        controller.neutral,
                        controller.L,
                        controller.r,
                    )

                now = time.perf_counter()

                # =============================================
                # 7. VIEWER SYNC ~60 Hz
                # =============================================

                if now >= next_viewer:
                    viewer.sync(state_only=True)
                    next_viewer = now + 1.0 / 60.0

                # =============================================
                # 8. OVERLAY ~15 Hz
                # =============================================

                if now >= next_overlay:
                    text = build_overlay(
                        controller,
                        keyboard_state,
                        measurements,
                        pcc,
                    )

                    with viewer.lock():
                        viewer.set_texts([
                            (
                                mujoco.mjtFontScale.mjFONTSCALE_100,
                                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                text,
                                "",
                            )
                        ])

                    next_overlay = now + 1.0 / 15.0

                # =============================================
                # 9. LIGHT CPU PACING
                # =============================================

                time.sleep(
                    max(
                        0.0002,
                        model.opt.timestep * 0.25,
                    )
                )

    finally:
        keyboard_state.clear()
        listener.stop()


if __name__ == "__main__":
    main()
