import numpy as np

def azimuth_elevation_to_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Convert azimuth/elevation to 3D unit vector.
    
    Uses geographical convention to match WAV file naming:
    - Azimuth: 0° = North (+Y), 90° = East (+X), 180° = South (-Y), 270° = West (-X)
    - Elevation: 0° = horizontal, 90° = straight up (+Z)
    """
    azimuth_radians = np.deg2rad(azimuth_deg)
    elevation_radians = np.deg2rad(elevation_deg)
    
    # Convert from geographical (0°=North) to mathematical (0°=+X)
    # Correct mapping: az_math = π/2 − az_geo
    azimuth_math_radians = np.pi/2 - azimuth_radians
    
    return np.array([
        np.cos(elevation_radians) * np.cos(azimuth_math_radians),  # X = East
        np.cos(elevation_radians) * np.sin(azimuth_math_radians),  # Y = North
        np.sin(elevation_radians)                                  # Z = Up
    ], dtype=float)

def angular_distance_deg(azimuth1_deg, elevation1_deg, azimuth2_deg, elevation2_deg) -> float:
    """Angular distance (deg) between two az/el points on the sphere."""
    a1 = np.deg2rad(azimuth1_deg); e1 = np.deg2rad(elevation1_deg)
    a2 = np.deg2rad(azimuth2_deg); e2 = np.deg2rad(elevation2_deg)
    v1 = np.array([np.cos(e1)*np.cos(a1), np.cos(e1)*np.sin(a1), np.sin(e1)])
    v2 = np.array([np.cos(e2)*np.cos(a2), np.cos(e2)*np.sin(a2), np.sin(e2)])
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))

def expected_pair_lag_samples(
    unit_direction: np.ndarray,
    mic_pos_m1: np.ndarray,
    mic_pos_m2: np.ndarray,
    sampling_rate_hz: float,
    speed_of_sound_mps: float
) -> float:
    """
    Expected TDOA (in samples) for baseline (m1->m2) for a plane wave from unit_direction.
    Positive value means signal arrives at m1 earlier than m2 per this baseline sign convention.
    """
    baseline_m = mic_pos_m2 - mic_pos_m1
    tdoa_seconds = np.dot(baseline_m, unit_direction) / speed_of_sound_mps
    return float(tdoa_seconds * sampling_rate_hz)