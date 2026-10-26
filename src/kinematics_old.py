"""
DOC-Oc continuum arm kinematics.

Four tendon, fully-constrained, piecewise-constant-curvature model.

Tendon arrangement:

             +Y
              |
              |
        -X ---O--- +X
              |
              |
             -Y

The formulation follows the standard tendon-driven
constant-curvature relationship:

    l_i = L - d * theta * cos(phi_i - alpha)

where:

    l_i     = tendon length
    L       = neutral/backbone length
    d       = tendon routing radius
    theta   = bending angle
    alpha   = bending direction
    phi_i   = tendon angular location

For the actual MuJoCo model we use the measured
distance between every pair of adjacent disks, giving
10 local PCC sections.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np


TENDON_ORDER = (
    "px",
    "nx",
    "py",
    "ny",
)


@dataclass
class PCCSection:
    length_m: float

    theta_x_rad: float
    theta_y_rad: float

    theta_rad: float
    direction_rad: float

    curvature_x: float
    curvature_y: float

    curvature: float


@dataclass
class PCCResult:
    length_m: float

    bending_angle_rad: float
    bending_angle_deg: float

    direction_rad: float
    direction_deg: float

    curvature_1_m: float

    curvature_x_1_m: float
    curvature_y_1_m: float

    sections: list[PCCSection]


# ---------------------------------------------------------------------
# Motor geometry
# ---------------------------------------------------------------------

def motor_step_displacement(
    spool_radius_m: float,
    step_angle_deg: float,
) -> float:
    """
    Linear tendon displacement produced by one motor step.
    """

    step_angle_rad = math.radians(step_angle_deg)

    return spool_radius_m * step_angle_rad


# ---------------------------------------------------------------------
# Four-tendon PCC inverse relationship
# ---------------------------------------------------------------------

def tendon_lengths_from_configuration(
    centerline_length_m: float,
    bending_angle_rad: float,
    bending_direction_rad: float,
    tendon_radius_m: float,
) -> Dict[str, float]:
    """
    Convert PCC configuration -> four tendon lengths.

    Exact fully-constrained constant-curvature relationship:

        l_i = L - d * theta * cos(phi_i - alpha)
    """

    L = float(centerline_length_m)
    theta = float(bending_angle_rad)
    alpha = float(bending_direction_rad)
    d = float(tendon_radius_m)

    phi = {
        "px": 0.0,
        "py": math.pi / 2.0,
        "nx": math.pi,
        "ny": 3.0 * math.pi / 2.0,
    }

    lengths = {}

    for name, angle in phi.items():

        lengths[name] = (
            L
            -
            d
            * theta
            * math.cos(angle - alpha)
        )

    return lengths


# ---------------------------------------------------------------------
# Four-tendon PCC inverse kinematics
# ---------------------------------------------------------------------

def configuration_from_tendon_lengths(
    tendon_lengths: Dict[str, float],
    tendon_radius_m: float,
) -> PCCResult:
    """
    Recover PCC configuration from four tendon lengths.

    The neutral-axis length is estimated from the mean of
    all four tendon lengths.

    Tendon deviations:

        Δl_i = L - l_i

    Then:

        Δl_px - Δl_nx = 2 d theta cos(alpha)

        Δl_py - Δl_ny = 2 d theta sin(alpha)

    Therefore:

        theta_x = (l_nx - l_px)/(2d)

        theta_y = (l_ny - l_py)/(2d)

    and:

        theta = sqrt(theta_x² + theta_y²)

        alpha = atan2(theta_y, theta_x)
    """

    px = float(tendon_lengths["px"])
    nx = float(tendon_lengths["nx"])
    py = float(tendon_lengths["py"])
    ny = float(tendon_lengths["ny"])

    d = float(tendon_radius_m)

    L = (
        px
        +
        nx
        +
        py
        +
        ny
    ) / 4.0

    if d <= 0.0:
        raise ValueError("tendon_radius_m must be positive")

    theta_x = (
        nx - px
    ) / (
        2.0 * d
    )

    theta_y = (
        ny - py
    ) / (
        2.0 * d
    )

    theta = math.hypot(
        theta_x,
        theta_y,
    )

    alpha = math.atan2(
        theta_y,
        theta_x,
    )

    if theta < 1e-12:

        alpha = 0.0

    curvature_x = (
        theta_x / L
        if L > 1e-12
        else 0.0
    )

    curvature_y = (
        theta_y / L
        if L > 1e-12
        else 0.0
    )

    curvature = math.hypot(
        curvature_x,
        curvature_y,
    )

    return PCCResult(
        length_m=L,

        bending_angle_rad=theta,
        bending_angle_deg=math.degrees(theta),

        direction_rad=alpha,
        direction_deg=math.degrees(alpha),

        curvature_1_m=curvature,

        curvature_x_1_m=curvature_x,
        curvature_y_1_m=curvature_y,

        sections=[],
    )


# ---------------------------------------------------------------------
# Piecewise PCC from actual MuJoCo disk positions
# ---------------------------------------------------------------------

def piecewise_pcc_from_segment_lengths(
    segment_lengths: np.ndarray,
    tendon_radius_m: float,
) -> PCCResult:
    """
    Calculate local PCC curvature from every disk interval.

    segment_lengths shape:

        (10, 4)

    Columns:

        px
        nx
        py
        ny

    Each row is the actual MuJoCo distance between two
    consecutive disk tendon sites.

    This is more accurate than using only total tendon lengths.
    """

    segment_lengths = np.asarray(
        segment_lengths,
        dtype=np.float64,
    )

    if segment_lengths.ndim != 2:
        raise ValueError(
            "segment_lengths must be a 2D array"
        )

    if segment_lengths.shape[1] != 4:
        raise ValueError(
            "segment_lengths must have four tendons"
        )

    d = float(tendon_radius_m)

    if d <= 0.0:
        raise ValueError(
            "tendon_radius_m must be positive"
        )

    sections = []

    total_theta_x = 0.0
    total_theta_y = 0.0
    total_length = 0.0

    for row in segment_lengths:

        px = row[0]
        nx = row[1]
        py = row[2]
        ny = row[3]

        # Neutral-axis length of this local section.
        L = (
            px
            +
            nx
            +
            py
            +
            ny
        ) / 4.0

        # Positive value means the corresponding tendon
        # is shorter than the neutral axis.
        theta_x = (
            nx - px
        ) / (
            2.0 * d
        )

        theta_y = (
            ny - py
        ) / (
            2.0 * d
        )

        theta = math.hypot(
            theta_x,
            theta_y,
        )

        direction = math.atan2(
            theta_y,
            theta_x,
        )

        curvature_x = (
            theta_x / L
            if L > 1e-12
            else 0.0
        )

        curvature_y = (
            theta_y / L
            if L > 1e-12
            else 0.0
        )

        curvature = math.hypot(
            curvature_x,
            curvature_y,
        )

        sections.append(
            PCCSection(
                length_m=L,

                theta_x_rad=theta_x,
                theta_y_rad=theta_y,

                theta_rad=theta,
                direction_rad=direction,

                curvature_x=curvature_x,
                curvature_y=curvature_y,

                curvature=curvature,
            )
        )

        total_theta_x += theta_x
        total_theta_y += theta_y
        total_length += L

    total_theta = math.hypot(
        total_theta_x,
        total_theta_y,
    )

    total_direction = math.atan2(
        total_theta_y,
        total_theta_x,
    )

    if total_theta < 1e-12:
        total_direction = 0.0

    total_curvature = (
        total_theta / total_length
        if total_length > 1e-12
        else 0.0
    )

    return PCCResult(
        length_m=total_length,

        bending_angle_rad=total_theta,
        bending_angle_deg=math.degrees(total_theta),

        direction_rad=total_direction,
        direction_deg=math.degrees(total_direction),

        curvature_1_m=total_curvature,

        curvature_x_1_m=(
            total_theta_x / total_length
            if total_length > 1e-12
            else 0.0
        ),

        curvature_y_1_m=(
            total_theta_y / total_length
            if total_length > 1e-12
            else 0.0
        ),

        sections=sections,
    )