import numpy as np
import librosa
from .config import AUDIO_CONFIG
from .features import extract_mfcc_features, feature_vector_rescaling


def peak_normalization(audio: np.ndarray, frame_size_ms: int = AUDIO_CONFIG["frame_duration_ms"], sr: int = AUDIO_CONFIG["sample_rate"]) -> np.ndarray:
    """Apply time-domain peak normalization per frame.
    Frames are `frame_size_ms` long with no overlap.
    """
    frame_size = int(frame_size_ms * sr / 1000)
    if frame_size <= 0:
        return audio

    normalized_frames = []
    for i in range(0, len(audio), frame_size):
        frame = audio[i : i + frame_size]
        if frame.size == 0:
            continue
        max_amplitude = np.max(np.abs(frame))
        if max_amplitude > 0:
            normalized_frame = frame / max_amplitude
        else:
            normalized_frame = frame
        normalized_frames.append(normalized_frame)

    if not normalized_frames:
        return audio

    return np.concatenate(normalized_frames)


def window_features(scaled_features: np.ndarray, time_steps: int = AUDIO_CONFIG["time_steps"]) -> np.ndarray:
    """Create sliding windows over time for LSTM input.

    Input: scaled_features shape (T, F)
    Output: windows shape (N, time_steps, F)
    """
    T = len(scaled_features)
    if T < time_steps:
        return np.empty((0, time_steps, scaled_features.shape[1]), dtype=scaled_features.dtype)

    windows = []
    for i in range(0, T - time_steps + 1):
        window = scaled_features[i : i + time_steps]
        windows.append(window)
    return np.asarray(windows)


def preprocess_audio_array(audio: np.ndarray, sr: int = AUDIO_CONFIG["sample_rate"]) -> np.ndarray:
    """Complete preprocessing pipeline given an audio array.

    1) Peak normalization
    2) MFCC extraction (focus 0-8 kHz)
    3) L2 rescaling per frame
    4) Window into `time_steps`
    """
    normalized_audio = peak_normalization(audio, sr=sr)
    mfcc_features = extract_mfcc_features(normalized_audio, sr=sr)
    scaled_features = feature_vector_rescaling(mfcc_features)
    windowed = window_features(scaled_features)
    return windowed


def preprocess_audio_file(audio_file_path: str, sr: int = AUDIO_CONFIG["sample_rate"]) -> np.ndarray:
    audio, file_sr = librosa.load(audio_file_path, sr=sr)
    return preprocess_audio_array(audio, sr=sr)
