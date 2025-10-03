from __future__ import annotations

import numpy as np
from typing import Tuple
from .build_steering_grid import build_steering_grid


def setup_srp_grid(
    azimuth_range_deg: Tuple[float, float],
    elevation_range_deg: Tuple[float, float],
    az_step_deg: float,
    el_step_deg: float,
):
    """
    Build SRP-PHAT search grids and steering vectors given ranges and step sizes.

    Returns (az_grid, el_grid, steering_vectors) where:
      - az_grid: 1D np.ndarray of azimuth degrees
      - el_grid: 1D np.ndarray of elevation degrees
      - steering_vectors: 2D np.ndarray of shape (len(el_grid)*len(az_grid), 3)
    """
    az0, az1 = float(azimuth_range_deg[0]), float(azimuth_range_deg[1])
    el0, el1 = float(elevation_range_deg[0]), float(elevation_range_deg[1])

    num_az = int(round((az1 - az0) / float(az_step_deg))) + 1
    num_el = int(round((el1 - el0) / float(el_step_deg))) + 1

    az_grid, el_grid, steering_vectors = build_steering_grid(
        azimuth_range_deg=(az0, az1),
        elevation_range_deg=(el0, el1),
        num_azimuth=num_az,
        num_elevation=num_el,
    )
    return az_grid, el_grid, steering_vectors


def setup_srp_phat_grid_context(
    azimuth_range_deg: Tuple[float, float],
    elevation_range_deg: Tuple[float, float],
    az_step_deg: float,
    el_step_deg: float,
):
    """
    Build SRP-PHAT grids and context.

    Returns: (az_grid, el_grid, steering_vectors, srp_power_map, summary_str)
    - srp_power_map is initialized to zeros with shape (len(el_grid), len(az_grid))
    - summary_str describes grid dimensions and step sizes
    """
    az_grid, el_grid, steering_vectors = setup_srp_grid(
        azimuth_range_deg=azimuth_range_deg,
        elevation_range_deg=elevation_range_deg,
        az_step_deg=az_step_deg,
        el_step_deg=el_step_deg,
    )
    srp_power_map = np.zeros((len(el_grid), len(az_grid)), dtype=float)
    summary = (
        f"\u2713 SRP-PHAT grid: {len(az_grid)} az \u00D7 {len(el_grid)} el = "
        f"{len(steering_vectors)} directions (step: {az_step_deg}° az, {el_step_deg}° el)"
    )
    return az_grid, el_grid, steering_vectors, srp_power_map, summary
