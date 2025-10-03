from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple
from scipy.io.wavfile import read as wav_read
from scipy import signal
import soundfile as sf


def _resample_if_needed(audio: np.ndarray, in_sr: int, target_sr: int) -> np.ndarray:
    if in_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    # Use resample_poly for quality and performance
    from math import gcd
    g = gcd(in_sr, target_sr)
    up, down = target_sr // g, in_sr // g
    chans = []
    for ch in range(audio.shape[1]):
        chans.append(signal.resample_poly(audio[:, ch], up, down).astype(np.float32))
    # Align lengths
    min_len = min(len(x) for x in chans)
    chans = [x[:min_len] for x in chans]
    return np.column_stack(chans).astype(np.float32)


def load_audio(audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load audio as float32 array with shape (samples, channels), resampled to target_sr.

    Returns (audio, sample_rate) where audio is float32 and sample_rate == target_sr.
    Tries scipy.io.wavfile first; falls back to soundfile for robust format handling.
    """
    p = str(audio_path)
    # Try scipy wavfile
    try:
        sr, audio = wav_read(p)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        audio = _resample_if_needed(audio, sr, target_sr)
        return audio, target_sr
    except Exception:
        pass
    # Fallback: soundfile (handles FLAC/OGG/etc.)
    data, sr = sf.read(p, always_2d=True, dtype='float32')  # shape (samples, channels)
    audio = data.astype(np.float32, copy=False)
    audio = _resample_if_needed(audio, sr, target_sr)
    return audio, target_sr
