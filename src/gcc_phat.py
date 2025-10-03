import numpy as np
# Backend that switches between NumPy (CPU) and CuPy (GPU)
from .nd_backend import xp, fft, hann_window, asarray, asnumpy

def compute_gcc_phat_singleblock(
    frame_channels: np.ndarray,   # Input: A NumPy array containing acoustic data for a single time frame.
                                  # Its shape is (num_mics, frame_length), meaning rows are for microphones
                                  # and columns represent samples within that frame.
    nfft: int,                    # Input: The number of FFT points to use.
    pair_m1_b=None,               # Optional backend array of first mic indices per pair (P,) to avoid per-call alloc
    pair_m2_b=None,               # Optional backend array of second mic indices per pair (P,)
) -> list[np.ndarray]:
    """
    Compute one GCC-PHAT curve per microphone pair using a single nfft window:
      - Take first nfft samples (zero-pad if shorter, truncate if longer).
      - Apply Hann window, rFFT, PHAT weighting, irFFT, fftshift.
    Returns: list of GCC arrays (length nfft), fftshifted so lags ~[-nfft/2, nfft/2 - 1].
    """
    # Extract dimensions from the input audio frame.
    num_mics, frame_length = frame_channels.shape # 'num_mics' is the total number of microphones in the array,
                                                  # and 'frame_length' is the number of audio samples in the current frame.

    # Generate all unique microphone pairs for cross-correlation.
    # GCC-PHAT is computed for every possible combination of two distinct microphones.
    mic_pairs = [(m1, m2) for m1 in range(num_mics) for m2 in range(m1+1, num_mics)] # This creates a list of tuples,
                                                                                    # e.g., for 4 mics: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).

    # Define a Hann window function (backend-specific: NumPy or CuPy).
    # Windowing helps reduce spectral leakage. Keep window on device if GPU.
    window = hann_window(nfft, sym=False)

    # Initialize an empty list to store the computed GCC-PHAT curves for each microphone pair.
    gcc_per_pair: list[np.ndarray] = [] # Each element in this list will be a NumPy array representing the GCC-PHAT for one pair.

    # Vectorized path: window and FFT once per mic
    # Slice/pad per mic to uniform nfft
    frame_len = frame_length if frame_length < nfft else nfft
    # Build (num_mics, nfft) with zero-pad if needed
    mic_time = np.zeros((num_mics, nfft), dtype=np.float32)
    mic_time[:, :frame_len] = frame_channels[:, :frame_len]
    mic_time_b = asarray(mic_time, dtype=xp.float32)
    xw_all = mic_time_b * window[None, :]
    # rFFT along time axis=1 -> (num_mics, nfreq)
    Xf_all = fft.rfft(xw_all, n=nfft, axis=1)

    # Build pair index arrays (use precomputed backend arrays if provided)
    if pair_m1_b is not None and pair_m2_b is not None:
        m1_b = pair_m1_b
        m2_b = pair_m2_b
    else:
        m1_idx = np.array([p[0] for p in mic_pairs], dtype=np.int32)
        m2_idx = np.array([p[1] for p in mic_pairs], dtype=np.int32)
        m1_b = asarray(m1_idx)
        m2_b = asarray(m2_idx)
    X1 = Xf_all[m1_b, :]
    X2 = Xf_all[m2_b, :]
    cross = X1 * xp.conj(X2)
    phat = cross / (xp.abs(cross) + 1e-12)
    # Batched irFFT along axis=1 -> (P, nfft)
    gcc_time_all = fft.irfft(phat, n=nfft, axis=1).real
    # Center zero-lag along axis=1
    try:
        gcc_shifted_all = xp.fft.fftshift(gcc_time_all, axes=1)
    except Exception:
        # Fallback: roll by nfft//2 if axes kw not supported
        gcc_shifted_all = xp.roll(gcc_time_all, shift=nfft//2, axis=1)

    # Return as list (keep backend arrays)
    for p in range(len(mic_pairs)):
        gcc_per_pair.append(gcc_shifted_all[p, :])

    # Return the list containing all GCC-PHAT curves, one for each microphone pair.
    return gcc_per_pair