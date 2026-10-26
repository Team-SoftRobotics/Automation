"""Measurement utilities for the DOC-Oc model."""

import numpy as np
import mujoco


def _site_position(model, data, name):
    sid = model.site(name).id
    return np.array(data.site_xpos[sid], dtype=float)


def _tendon_lengths(model, data):
    return {
        name: float(data.ten_length[model.tendon(f"tendon_{name}").id])
        for name in ("px", "nx", "py", "ny")
    }


def _nitinol_lengths(model, data):
    return np.array([
        float(data.ten_length[model.tendon(f"central_wire_{i}").id])
        for i in range(1, 8)
    ], dtype=float)


def get_measurements(model, data, nitinol_reference_lengths=None):
    tip = _site_position(model, data, "tip")
    tendon_lengths = _tendon_lengths(model, data)
    nitinol_lengths = _nitinol_lengths(model, data)

    distance_xy = float(np.linalg.norm(tip[:2]))

    if nitinol_reference_lengths is None:
        nitinol_strains = np.zeros(7, dtype=float)
    else:
        ref = np.asarray(nitinol_reference_lengths, dtype=float)
        nitinol_strains = (nitinol_lengths - ref) / np.maximum(np.abs(ref), 1e-12)

    # Only tensile strain counts for the Nitinol warning.
    tensile = np.maximum(nitinol_strains, 0.0)
    max_tensile = float(np.max(tensile))

    return {
        "tip_position_m": tip,
        "tip_distance_xy_m": distance_xy,
        "tendon_lengths_m": tendon_lengths,
        "nitinol_lengths_m": nitinol_lengths,
        "nitinol_strains": nitinol_strains,
        "max_nitinol_tensile_strain": max_tensile,
    }
