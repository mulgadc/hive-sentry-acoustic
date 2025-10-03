import numpy as np
import librosa
from .config import AUDIO_CONFIG


def frequency_warping(audio: np.ndarray, sr: int = AUDIO_CONFIG["sample_rate"]) -> list:
    """Apply frequency warping by resampling around alpha in [0.8, 1.2].

    Returns a list of augmented audio arrays, each re-sampled back to `sr`.
    """
    augmented_data = []
    alpha_values = np.arange(0.8, 1.21, 0.02)  # 21 values

    for alpha in alpha_values:
        new_sr = int(sr * alpha)
        warped_audio = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
        final_audio = librosa.resample(warped_audio, orig_sr=new_sr, target_sr=sr)
        augmented_data.append(final_audio.astype(np.float32))

    return augmented_data
