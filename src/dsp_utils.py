from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy import signal


def make_pair_indices(num_mics: int) -> List[Tuple[int, int]]:
    """Return all unique mic index pairs (i<j)."""
    return [(i, j) for i in range(num_mics) for j in range(i + 1, num_mics)]


def compute_levels(frame: np.ndarray, num_series: int = 5) -> List[float]:
    """Per-channel RMS (std) for the first num_series channels of the frame (samples x channels)."""
    vals = [float(np.std(frame[:, i])) for i in range(min(num_series, frame.shape[1]))]
    while len(vals) < num_series:
        vals.append(0.0)
    return vals


def compute_spec_power_slice(sig: np.ndarray, sample_rate: float, spec_nfft: int) -> np.ndarray:
    """Compute magnitude^2 of the last column of a Hann spectrogram for a 1-D signal.
    Returns a (spec_nfft/2+1,) float32 vector.
    """
    if sig.size < spec_nfft:
        sig = np.pad(sig, (spec_nfft - sig.size, 0), mode='constant')
    f, t_arr, Sxx = signal.spectrogram(
        sig.astype(np.float32, copy=False),
        fs=sample_rate,
        nperseg=spec_nfft,
        noverlap=0,
        window='hann',
        mode='magnitude',
    )
    if Sxx.shape[1] == 0:
        return np.zeros((len(f),), dtype=np.float32)
    return (Sxx[:, -1] ** 2).astype(np.float32)


def normalize_map(power_map: np.ndarray, gamma: float) -> tuple[np.ndarray, float, float]:
    """Normalize a 2D map to 0..1 with optional gamma; also return (min,max) of the input."""
    pmin = float(np.min(power_map))
    pmax = float(np.max(power_map))
    contrast = max(1e-12, pmax - pmin)
    norm = (power_map - pmin) / contrast
    if gamma is not None and gamma > 0:
        norm = np.power(norm, float(gamma))
    return norm.astype(np.float32), pmin, pmax
