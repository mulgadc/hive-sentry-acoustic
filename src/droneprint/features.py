import numpy as np
import librosa
from sklearn.preprocessing import normalize
from .config import AUDIO_CONFIG


def extract_mfcc_features(audio: np.ndarray, sr: int = AUDIO_CONFIG["sample_rate"]) -> np.ndarray:
    """Extract MFCC features focused on 0-8 kHz.
    Returns shape (T, n_mfcc)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=AUDIO_CONFIG["n_mfcc"],
        n_fft=2048,
        hop_length=512,
        fmax=AUDIO_CONFIG["fmax"],
    )
    return mfcc.T.astype(np.float32)


def feature_vector_rescaling(mfcc_features: np.ndarray) -> np.ndarray:
    """Apply L2 normalization to feature vectors along time axis (rows)."""
    return normalize(mfcc_features, norm="l2", axis=1)
