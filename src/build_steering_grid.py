import numpy as np
from .helpers import azimuth_elevation_to_unit_vector

def build_steering_grid(
    azimuth_range_deg=(0.0, 360.0),
    elevation_range_deg=(0.0, 90.0),
    num_azimuth=181,
    num_elevation=31
):
    """Create az/el grids and corresponding plane-wave unit vectors (row-major by elevation, then azimuth)."""
    azimuth_grid_deg = np.linspace(*azimuth_range_deg, num=num_azimuth)
    elevation_grid_deg = np.linspace(*elevation_range_deg, num=num_elevation)
    steering_unit_vectors = np.array([
        azimuth_elevation_to_unit_vector(az_deg, el_deg)
        for el_deg in elevation_grid_deg
        for az_deg in azimuth_grid_deg
    ], dtype=float)
    return azimuth_grid_deg, elevation_grid_deg, steering_unit_vectors