import numpy as np
from .gcc_phat import compute_gcc_phat_singleblock
from .helpers import expected_pair_lag_samples
# Backend that switches between NumPy (CPU) and CuPy (GPU)
from .nd_backend import xp, asarray, asnumpy, take_along_axis, clip

def compute_srp_phat_for_frame(
    frame_channels: np.ndarray,        # Input: Acoustic data for one time frame with shape (num_mics, frame_length).
                                      # Each row contains audio samples from one microphone for the current frame.
    mic_positions_m: np.ndarray,       # Input: 3D positions of microphones in meters with shape (num_mics, 3).
                                      # Each row is [x, y, z] coordinates of one microphone in the array.
    azimuth_grid_deg: np.ndarray,     # Input: 1D array of azimuth angles (in degrees) to search over.
                                      # These define the horizontal angular search space (e.g., 0° to 360°).
    elevation_grid_deg: np.ndarray,   # Input: 1D array of elevation angles (in degrees) to search over.
                                      # These define the vertical angular search space (e.g., -90° to +90°).
    steering_unit_vectors: np.ndarray, # Input: Pre-computed unit direction vectors with shape (num_directions, 3).
                                      # Each row is a unit vector [x, y, z] pointing toward one search direction.
                                      # Total directions = len(elevation_grid_deg) * len(azimuth_grid_deg).
    sampling_rate_hz: float,          # Input: Audio sampling rate in Hz, used to convert time delays to sample indices.
    speed_of_sound_mps: float = 343.0, # Input: Speed of sound in meters per second (typically ~343 m/s).
    nfft: int = 1024                  # Input: FFT length used for GCC-PHAT computation, determines lag resolution.
):
    """
    Steered Response Power with Phase Transform (SRP-PHAT) for Direction-of-Arrival estimation.
    
    This function implements the core SRP-PHAT beamforming algorithm:
    P(q) = sum_{l<k} R_lk(Δ_lk(q))
    
    Where:
    - P(q) is the steered power for direction q
    - R_lk(·) is the GCC-PHAT correlation function between microphones l and k  
    - Δ_lk(q) is the expected time delay between mics l,k for source direction q
    
    The algorithm searches all possible source directions by:
    1. Computing GCC-PHAT curves for all microphone pairs
    2. For each candidate direction, calculating expected inter-mic delays
    3. Sampling GCC-PHAT curves at those expected delays and summing
    4. The direction with maximum summed response is the estimated source location
    
    Returns: GCC curves, best/next DOA estimates, powers, and the full 2D response map.
    """
    
    # LOG: Input parameters
    print(f"[SRP_PHAT] Function called with:")
    print(f"  - Frame shape: {frame_channels.shape}")
    print(f"  - Azimuth grid: {len(azimuth_grid_deg)} points ({azimuth_grid_deg[0]:.1f}° to {azimuth_grid_deg[-1]:.1f}°)")
    print(f"  - Elevation grid: {len(elevation_grid_deg)} points ({elevation_grid_deg[0]:.1f}° to {elevation_grid_deg[-1]:.1f}°)")
    print(f"  - Steering vectors shape: {steering_unit_vectors.shape}")
    print(f"  - Expected directions: {len(elevation_grid_deg) * len(azimuth_grid_deg)}")
    
    # Step 1: Compute GCC-PHAT correlation curves for all microphone pairs.
    # This gives us R_lk(τ) - the correlation function showing how similar
    # each pair's signals are at different time delays τ.
    gcc_per_pair = compute_gcc_phat_singleblock(frame_channels, nfft=nfft)
    
    # Extract array geometry information for the steering calculations.
    num_mics = mic_positions_m.shape[0]  # Total number of microphones in the array.
    
    # Generate all unique microphone pairs for processing.
    # We only need pairs (m1, m2) where m1 < m2 to avoid duplicate calculations.
    mic_pairs = [(m1, m2) for m1 in range(num_mics) for m2 in range(m1+1, num_mics)]
    
    # Initialize the steered response power array.
    # This will store P(q) - the accumulated correlation power for each search direction q.
    power = xp.zeros(steering_unit_vectors.shape[0], dtype=float)
    
    # Calculate the center index for the fftshifted GCC-PHAT curves.
    # After fftshift, zero-lag (no delay) corresponds to index nfft//2.
    # Negative lags are to the left, positive lags to the right of center.
    center = nfft // 2  
    
    # Step 2: Steering loop - evaluate response power for each candidate direction.
    for q, unit_dir in enumerate(steering_unit_vectors):  # q is the direction index,
                                                         # unit_dir is the 3D unit vector [x,y,z] for this direction.
        
        # Initialize accumulator on backend to avoid implicit host conversions
        acc = xp.asarray(0.0, dtype=power.dtype)
        
        # Step 3: Sum contributions from all microphone pairs for this direction.
        for pair_index, (m1, m2) in enumerate(mic_pairs):
            
            # Calculate the expected time-difference-of-arrival (TDOA) between 
            # microphones m1 and m2 for a source arriving from direction unit_dir.
            # This uses array geometry and acoustic propagation physics.
            tau_samples = expected_pair_lag_samples(
                unit_dir,                    # Source direction as unit vector
                mic_positions_m[m1],        # Position of first microphone  
                mic_positions_m[m2],        # Position of second microphone
                sampling_rate_hz,           # Convert time delay to sample units
                speed_of_sound_mps          # Acoustic propagation speed
            )
            
            # Convert the expected delay to an array index in the GCC-PHAT curve.
            # Round to nearest integer sample and offset by center for fftshift indexing.
            # Convert device scalar to Python int explicitly to avoid implicit CuPy->NumPy conversion
            idx = int(asnumpy(xp.round(tau_samples))) + center
            
            # Sample the GCC-PHAT curve at the expected delay index.
            # Only include valid indices to avoid array bounds errors.
            if 0 <= idx < nfft:
                acc = acc + gcc_per_pair[pair_index][idx]  # Add this pair's correlation contribution
                                                     # at the expected delay for direction q.
        
        # Store the total accumulated power for this direction.
        # Higher values indicate better correlation alignment = more likely source direction.
        power[q] = acc
    
    # Step 4: Find the best (strongest) and second-best directions.
    # Sort all directions by their response power in descending order.
    order = xp.argsort(power)[::-1]  # Indices sorted by decreasing power
    # Convert device scalars to Python ints explicitly
    q_best = int(asnumpy(order[0]))
    q_next = int(asnumpy(order[1]))
    
    # Step 5: Convert the best direction indices back to azimuth/elevation angles.
    # The steering grid is organized as a 2D array: elevation rows × azimuth columns.
    num_az = len(azimuth_grid_deg)  # Number of azimuth angles in the search grid
    
    # Convert flat index to 2D grid coordinates (elevation_idx, azimuth_idx).
    el_idx_best, az_idx_best = q_best // num_az, q_best % num_az    # Best direction grid coordinates
    el_idx_next, az_idx_next = q_next // num_az, q_next % num_az    # Second-best direction grid coordinates
    
    # Extract the actual angle values from the search grids.
    best_azimuth_deg = float(azimuth_grid_deg[az_idx_best])      # Best azimuth angle in degrees
    best_elevation_deg = float(elevation_grid_deg[el_idx_best])   # Best elevation angle in degrees  
    next_azimuth_deg = float(azimuth_grid_deg[az_idx_next])      # Second-best azimuth angle in degrees
    next_elevation_deg = float(elevation_grid_deg[el_idx_next])   # Second-best elevation angle in degrees
    
    # Step 6: Reshape the 1D power array into a 2D response map for visualization.
    # This creates a heatmap showing response power vs. elevation (rows) and azimuth (columns).
    print(f"[SRP_PHAT] Reshaping power array:")
    print(f"  - Input power array shape: {power.shape}")
    print(f"  - Grid dimensions: elevation={len(elevation_grid_deg)}, azimuth={len(azimuth_grid_deg)}")
    print(f"  - Expected total: {len(elevation_grid_deg) * len(azimuth_grid_deg)}")
    power_map = asnumpy(power.reshape(len(elevation_grid_deg), len(azimuth_grid_deg)))
    print(f"  - Reshaped power_map shape: {power_map.shape}")
    print(f"  - Power value range: {power_map.min():.3f} to {power_map.max():.3f}")
    
    # Return all computed results for further analysis and visualization.
    return (
        gcc_per_pair,          # List of GCC-PHAT correlation curves for each microphone pair
        best_azimuth_deg,      # Estimated source azimuth (primary detection) in degrees
        best_elevation_deg,    # Estimated source elevation (primary detection) in degrees  
        next_azimuth_deg,      # Second-best azimuth estimate in degrees
        next_elevation_deg,    # Second-best elevation estimate in degrees
        float(asnumpy(power[q_best])),  # Response power value for the best direction
        float(asnumpy(power[q_next])),  # Response power value for the second-best direction  
        power_map              # 2D array of response power vs. elevation and azimuth for visualization
    )

def compute_srp_phat_windowed_max(
    gcc_curves: list[np.ndarray],
    mic_positions_m: np.ndarray,
    pair_indices: list[tuple[int, int]],
    steering_unit_vectors: np.ndarray,
    azimuth_grid_deg: np.ndarray,
    elevation_grid_deg: np.ndarray,
    sampling_rate_hz: float,
    speed_of_sound_mps: float = 343.0,
    nfft: int = 1024,
    search_window: int = 5,
    pre_idx_all: np.ndarray | None = None,
    pre_offsets: np.ndarray | None = None,
    pre_idx_pw: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """
    SRP-PHAT using windowed maximum around expected TDOA per (pair, direction),
    optimized to reuse precomputed GCC curves.

    Returns (power_map, best_az_deg, best_el_deg, best_power).
    """
    # Shapes
    Q = steering_unit_vectors.shape[0]
    P = len(pair_indices)
    center = nfft // 2

    # Stack GCC curves to ndarray of shape (P, nfft)
    # Move GCC curves to backend array (GPU if available)
    # IMPORTANT: Use backend stack to avoid implicit CuPy->NumPy conversion.
    # gcc_curves elements are backend arrays already (NumPy or CuPy). Using np.stack
    # on a list of CuPy arrays triggers CuPy's implicit conversion error. xp.stack
    # keeps the operation on the active backend device.
    gcc_arr = xp.stack(gcc_curves, axis=0)  # (P, nfft)

    # Compute (or reuse) expected-delay indices for all (pair, dir): (P, Q)
    if pre_idx_all is None:
        mp = asarray(np.asarray(mic_positions_m, dtype=float))
        # Build baselines (P,3) on backend
        baselines = asarray(np.array([
            mp[m2] - mp[m1]
            for (m1, m2) in pair_indices
        ], dtype=float))  # (P, 3)
        sv = asarray(steering_unit_vectors)
        proj = baselines @ sv.T  # (P, Q)
        tau_samples_all = (proj / float(speed_of_sound_mps)) * float(sampling_rate_hz)
        idx_all = xp.rint(tau_samples_all).astype(xp.int32) + center  # (P, Q)
    else:
        idx_all = asarray(pre_idx_all)  # (P, Q)

    # Build (or reuse) window offsets
    if pre_offsets is None:
        offsets = asarray(np.arange(-int(search_window), int(search_window) + 1, dtype=np.int32))  # (W,)
    else:
        offsets = asarray(pre_offsets)  # (W,)

    # Vectorized gather across pairs and window
    # idx_pw: (P, Q, W)
    if pre_idx_pw is not None:
        # Use precomputed backend indices
        idx_pw = asarray(pre_idx_pw)
    else:
        idx_pw = idx_all[:, :, None] + offsets[None, None, :]
        clip(idx_pw, 0, nfft - 1, out=idx_pw)
    # Expand gcc_arr to (P, 1, nfft) for take_along_axis
    gcc_exp = gcc_arr[:, None, :]
    vals = take_along_axis(gcc_exp, idx_pw, axis=2)  # (P, Q, W)
    # Window max per (P,Q), then sum across P -> (Q,)
    power = vals.max(axis=2).sum(axis=0)

    # Best direction (convert device scalar to Python int explicitly)
    best_idx = int(asnumpy(xp.argmax(power)))
    num_az = len(azimuth_grid_deg)
    el_idx, az_idx = divmod(best_idx, num_az)
    best_az = float(azimuth_grid_deg[az_idx])
    best_el = float(elevation_grid_deg[el_idx])
    # Convert device scalar to Python float if needed
    best_power = float(asnumpy(power[best_idx]))

    # Map shape (el x az)
    power_map = asnumpy(power.reshape(len(elevation_grid_deg), len(azimuth_grid_deg)))

    return power_map, best_az, best_el, best_power