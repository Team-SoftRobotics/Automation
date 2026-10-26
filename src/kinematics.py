"""
DOC-OC Continuum Arm
Tendon-space and constant-curvature kinematics.

Four actuation tendons:
    px = +X side
    nx = -X side
    py = +Y side
    ny = -Y side

Convention:
    W -> +Y
    S -> -Y
    A -> -X
    D -> +X
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PCCResult:
    """
    Result of four-tendon constant-curvature estimation.
    """

    # Curvature components
    curvature_x_1_m: float
    curvature_y_1_m: float

    # Total curvature
    curvature_1_m: float

    # Total bending angle
    bending_angle_rad: float
    bending_angle_deg: float

    # Bending direction in XY plane
    direction_rad: float
    direction_deg: float

    # Common axial strain estimated from all four tendons
    axial_strain: float


# ============================================================
# TENDON GEOMETRY
# ============================================================

TENDON_NAMES = (
    "px",
    "nx",
    "py",
    "ny",
)


TENDON_ANGLES_RAD = np.array(
    [
        0.0,
        math.pi,
        math.pi / 2.0,
        -math.pi / 2.0,
    ],
    dtype=float,
)


# ============================================================
# MOTOR STEP
# ============================================================

def motor_step_displacement(
    spool_radius_m: float,
    step_angle_deg: float,
) -> float:
    """
    Convert one motor angular step into linear tendon travel.

    s = r * theta

    Parameters
    ----------
    spool_radius_m:
        Effective spool radius in metres.

    step_angle_deg:
        Motor step angle in degrees.

    Returns
    -------
    float
        Tendon displacement in metres.
    """

    if spool_radius_m <= 0.0:
        raise ValueError(
            "spool_radius_m must be positive."
        )

    theta_rad = math.radians(
        float(step_angle_deg)
    )

    return float(
        spool_radius_m * theta_rad
    )


# ============================================================
# BUILD TENDON JACOBIAN
# ============================================================

def build_tendon_jacobian(
    L: float,
    r: float,
) -> np.ndarray:
    """
    Construct the four-tendon PCC Jacobian.

    Model:

        Δl_i =
            -L*r*(kx*cos(phi_i) + ky*sin(phi_i))
            + L*eps

    Unknown vector:

        [kx, ky, eps]

    where:

        kx = curvature about the Y-related bending plane
        ky = curvature about the X-related bending plane
        eps = common axial strain

    Returns
    -------
    ndarray
        4 x 3 Jacobian.
    """

    if L <= 0.0:
        raise ValueError(
            "Continuum length L must be positive."
        )

    if r <= 0.0:
        raise ValueError(
            "Tendon routing radius r must be positive."
        )

    phi = TENDON_ANGLES_RAD

    J = np.column_stack(
        [
            -L * r * np.cos(phi),
            -L * r * np.sin(phi),
            np.full(4, L, dtype=float),
        ]
    )

    return J


# ============================================================
# SOLVE PCC
# ============================================================

def solve_pcc_from_tendon_lengths(
    tendon_lengths,
    neutral_lengths,
    L: float,
    r: float,
) -> PCCResult:
    """
    Estimate continuum curvature from measured tendon lengths.

    Four tendon lengths are used simultaneously:

        px
        nx
        py
        ny

    The neutral length is subtracted first.

    This avoids relying on a single antagonistic pair and
    gives a least-squares estimate using all four tendons.

    Parameters
    ----------
    tendon_lengths:
        Mapping containing current tendon lengths in metres.

    neutral_lengths:
        Mapping containing neutral tendon lengths in metres.

    L:
        Effective continuum length in metres.

    r:
        Tendon routing radius in metres.

    Returns
    -------
    PCCResult
    """

    # --------------------------------------------------------
    # Validate tendon inputs
    # --------------------------------------------------------

    for name in TENDON_NAMES:

        if name not in tendon_lengths:
            raise KeyError(
                f"Missing current tendon length: {name}"
            )

        if name not in neutral_lengths:
            raise KeyError(
                f"Missing neutral tendon length: {name}"
            )

    # --------------------------------------------------------
    # Validate geometry
    # --------------------------------------------------------

    if L <= 0.0:
        raise ValueError(
            "L must be positive."
        )

    if r <= 0.0:
        raise ValueError(
            "r must be positive."
        )

    # --------------------------------------------------------
    # Tendon length changes
    # --------------------------------------------------------

    delta_l = np.array(
        [
            float(tendon_lengths[name])
            - float(neutral_lengths[name])
            for name in TENDON_NAMES
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------

    J = build_tendon_jacobian(
        L=L,
        r=r,
    )

    # --------------------------------------------------------
    # Least-squares solution
    #
    # J @ [kx, ky, eps] = delta_l
    # --------------------------------------------------------

    solution, _, _, _ = np.linalg.lstsq(
        J,
        delta_l,
        rcond=None,
    )

    kx = float(solution[0])
    ky = float(solution[1])
    axial_strain = float(solution[2])

    # --------------------------------------------------------
    # Total curvature
    # --------------------------------------------------------

    curvature = math.hypot(
        kx,
        ky,
    )

    # --------------------------------------------------------
    # Bending angle
    #
    # theta = kappa * L
    # --------------------------------------------------------

    bending_angle_rad = (
        curvature * L
    )

    bending_angle_deg = math.degrees(
        bending_angle_rad
    )

    # --------------------------------------------------------
    # Bending direction
    #
    # +X = 0 deg
    # +Y = +90 deg
    # -X = ±180 deg
    # -Y = -90 deg
    # --------------------------------------------------------

    if curvature > 1e-12:

        direction_rad = math.atan2(
            ky,
            kx,
        )

    else:

        direction_rad = 0.0

    direction_deg = math.degrees(
        direction_rad
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return PCCResult(
        curvature_x_1_m=kx,
        curvature_y_1_m=ky,
        curvature_1_m=curvature,
        bending_angle_rad=bending_angle_rad,
        bending_angle_deg=bending_angle_deg,
        direction_rad=direction_rad,
        direction_deg=direction_deg,
        axial_strain=axial_strain,
    )


# ============================================================
# PCC TIP POSITION
# ============================================================

def pcc_tip_from_curvature(
    kx: float,
    ky: float,
    L: float,
) -> np.ndarray:
    """
    Calculate the continuum tip position using constant
    curvature.

    Base frame:

        X = horizontal
        Y = horizontal
        Z = undeformed arm axis

    For zero curvature:

        tip = [0, 0, L]

    For non-zero curvature:

        theta = kappa * L

        rho = 1 / kappa

    Returns
    -------
    ndarray
        [x, y, z] tip position in metres.
    """

    if L <= 0.0:
        raise ValueError(
            "L must be positive."
        )

    kappa = math.hypot(
        float(kx),
        float(ky),
    )


    if kappa < 1e-12:

        return np.array(
            [
                0.0,
                0.0,
                L,
            ],
            dtype=float,
        )


    theta = kappa * L

    ux = float(kx) / kappa
    uy = float(ky) / kappa

    rho = 1.0 / kappa

    x = (
        rho
        * (1.0 - math.cos(theta))
        * ux
    )

    y = (
        rho
        * (1.0 - math.cos(theta))
        * uy
    )

    z = (
        rho
        * math.sin(theta)
    )

    return np.array(
        [
            x,
            y,
            z,
        ],
        dtype=float,
    )


def tendon_lengths_to_tip(
    tendon_lengths,
    neutral_lengths,
    L: float,
    r: float,
):
    """
    Convenience function.

    Converts four measured tendon lengths directly into:

        PCCResult
        tip_position

    Returns
    -------
    PCCResult, ndarray
    """

    result = solve_pcc_from_tendon_lengths(
        tendon_lengths=tendon_lengths,
        neutral_lengths=neutral_lengths,
        L=L,
        r=r,
    )

    tip = pcc_tip_from_curvature(
        kx=result.curvature_x_1_m,
        ky=result.curvature_y_1_m,
        L=L,
    )

    return result, tip