import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from .config import AUDIO_CONFIG
from .preprocessing import preprocess_audio_array
from .augmentation import frequency_warping


@dataclass
class DatasetPaths:
    root: str
    ds1: str
    ds2: str
    ds3: str


def discover_dataset_paths(datasets_root: str) -> DatasetPaths:
    ds1 = os.path.join(datasets_root, "DS1_drone_sounds")
    ds2 = os.path.join(datasets_root, "DS2_background")
    ds3 = os.path.join(datasets_root, "DS3_unknown")
    return DatasetPaths(root=datasets_root, ds1=ds1, ds2=ds2, ds3=ds3)


def list_audio_files(root: str, split: str) -> List[str]:
    """Recursively list audio files under `root/<split>/.../*.wav`"""
    pattern_lower = os.path.join(root, split, "**", "*.wav")
    pattern_upper = os.path.join(root, split, "**", "*.WAV")
    files = glob.glob(pattern_lower, recursive=True) + glob.glob(pattern_upper, recursive=True)
    return sorted(files)


def load_audio(path: str, sr: int = AUDIO_CONFIG["sample_rate"]) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr)
    return audio.astype(np.float32)


def list_noise_files(paths: DatasetPaths, split: str) -> List[str]:
    pattern_lower = os.path.join(paths.ds2, split, "**", "*.wav")
    pattern_upper = os.path.join(paths.ds2, split, "**", "*.WAV")
    files = glob.glob(pattern_lower, recursive=True) + glob.glob(pattern_upper, recursive=True)
    return sorted(files)


def mix_to_snr(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Mix noise into signal to achieve target SNR in dB."""
    # Trim/pad noise to signal length
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(signal)]

    Ps = np.mean(signal ** 2) + 1e-12
    Pn = np.mean(noise ** 2) + 1e-12
    desired_Pn = Ps / (10 ** (target_snr_db / 10.0))
    scale = np.sqrt(desired_Pn / Pn)
    mixed = signal + noise * scale
    # Prevent clipping
    max_abs = np.max(np.abs(mixed)) + 1e-12
    if max_abs > 1.0:
        mixed = mixed / max_abs
    return mixed.astype(np.float32)


def build_binary_dataset(
    paths: DatasetPaths,
    split: str,
    augment: bool = False,
    noise_mix_pct: float = 0.0,
    noise_snr_choices: Tuple[float, ...] = (1.0, 3.0, 10.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary: Drone vs Non-Drone.
    DS1 -> label 1 (drone)
    DS2 -> label 0 (non-drone)
    Returns X shape (N, time_steps, n_mfcc) and y one-hot (N, 2)
    """
    X_list, y_list = [], []

    # Drone (1)
    ds2_noises = list_noise_files(paths, split) if noise_mix_pct > 0 else []

    for fp in list_audio_files(paths.ds1, split):
        audio = load_audio(fp)
        # optional noise mix-in
        if ds2_noises and np.random.rand() < noise_mix_pct:
            nfp = np.random.choice(ds2_noises)
            noise = load_audio(nfp)
            snr = float(np.random.choice(noise_snr_choices))
            audio = mix_to_snr(audio, noise, snr)
        base_audios = [audio]
        if augment:
            base_audios.extend(frequency_warping(audio))
        for a in base_audios:
            windows = preprocess_audio_array(a)
            for w in windows:
                X_list.append(w)
                y_list.append(1)

    # Non-Drone (0)
    for fp in list_audio_files(paths.ds2, split):
        audio = load_audio(fp)
        base_audios = [audio]
        if augment:
            base_audios.extend(frequency_warping(audio))
        for a in base_audios:
            windows = preprocess_audio_array(a)
            for w in windows:
                X_list.append(w)
                y_list.append(0)

    if not X_list:
        return np.empty((0, AUDIO_CONFIG["time_steps"], AUDIO_CONFIG["n_mfcc"])) , np.empty((0, 2))

    X = np.asarray(X_list, dtype=np.float32)
    y = to_categorical(np.asarray(y_list, dtype=np.int32), num_classes=2)
    return X, y


def build_make_dataset(
    paths: DatasetPaths,
    split: str,
    augment: bool = False,
    noise_mix_pct: float = 0.0,
    noise_snr_choices: Tuple[float, ...] = (1.0, 3.0, 10.0),
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Make identification: label is top-level subfolder under DS1/<split>/
    Assumes structure DS1/<split>/<make>/<model>/*.wav (model subfolder optional).
    """
    X_list, y_list, makes = [], [], []

    make_dirs = [d for d in glob.glob(os.path.join(paths.ds1, split, "*")) if os.path.isdir(d)]
    ds2_noises = list_noise_files(paths, split) if noise_mix_pct > 0 else []
    for make_dir in sorted(make_dirs):
        make = os.path.basename(make_dir)
        makes.append(make)
        files = sorted(
            set(
                glob.glob(os.path.join(make_dir, "**", "*.wav"), recursive=True)
                + glob.glob(os.path.join(make_dir, "**", "*.WAV"), recursive=True)
            )
        )
        for fp in files:
            audio = load_audio(fp)
            if ds2_noises and np.random.rand() < noise_mix_pct:
                nfp = np.random.choice(ds2_noises)
                noise = load_audio(nfp)
                snr = float(np.random.choice(noise_snr_choices))
                audio = mix_to_snr(audio, noise, snr)
            base_audios = [audio]
            if augment:
                base_audios.extend(frequency_warping(audio))
            for a in base_audios:
                windows = preprocess_audio_array(a)
                for w in windows:
                    X_list.append(w)
                    y_list.append(make)

    if not y_list:
        le = LabelEncoder().fit([])
        X = np.empty((0, AUDIO_CONFIG["time_steps"], AUDIO_CONFIG["n_mfcc"]))
        y = np.empty((0, 0))
        return X, y, le
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_list)
    y = to_categorical(y_encoded, num_classes=len(le.classes_))
    X = np.asarray(X_list, dtype=np.float32)
    return X, y, le


def build_model_datasets_per_make(
    paths: DatasetPaths,
    split: str,
    augment: bool = False,
    noise_mix_pct: float = 0.0,
    noise_snr_choices: Tuple[float, ...] = (1.0, 3.0, 10.0),
) -> Dict[str, Tuple[np.ndarray, np.ndarray, LabelEncoder]]:
    """Return dict of make -> (X, y, label_encoder) for model identification per make.
    Label is the model subfolder under each make.
    """
    result: Dict[str, Tuple[np.ndarray, np.ndarray, LabelEncoder]] = {}
    make_dirs = [d for d in glob.glob(os.path.join(paths.ds1, split, "*")) if os.path.isdir(d)]

    ds2_noises = list_noise_files(paths, split) if noise_mix_pct > 0 else []
    for make_dir in sorted(make_dirs):
        make = os.path.basename(make_dir)
        model_dirs = [d for d in glob.glob(os.path.join(make_dir, "*")) if os.path.isdir(d)]
        X_list, y_list = [], []
        for model_dir in sorted(model_dirs):
            model_name = os.path.basename(model_dir)
            files = sorted(
                set(
                    glob.glob(os.path.join(model_dir, "*.wav"))
                    + glob.glob(os.path.join(model_dir, "*.WAV"))
                )
            )
            for fp in files:
                audio = load_audio(fp)
                if ds2_noises and np.random.rand() < noise_mix_pct:
                    nfp = np.random.choice(ds2_noises)
                    noise = load_audio(nfp)
                    snr = float(np.random.choice(noise_snr_choices))
                    audio = mix_to_snr(audio, noise, snr)
                base_audios = [audio]
                if augment:
                    base_audios.extend(frequency_warping(audio))
                for a in base_audios:
                    windows = preprocess_audio_array(a)
                    for w in windows:
                        X_list.append(w)
                        y_list.append(model_name)
        if not y_list:
            le = LabelEncoder().fit([])
            X = np.empty((0, AUDIO_CONFIG["time_steps"], AUDIO_CONFIG["n_mfcc"]))
            y = np.empty((0, 0))
            result[make] = (X, y, le)
        else:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_list)
            y = to_categorical(y_encoded, num_classes=len(le.classes_))
            X = np.asarray(X_list, dtype=np.float32)
            result[make] = (X, y, le)

    return result


def build_all_drone_models_dataset(paths: DatasetPaths, split: str, augment: bool = False) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Build closed-set dataset of all drone models in DS1 by model label.
    Structure assumed DS1/<split>/<make>/<model>/*.wav
    Returns X (N, T, F), y one-hot, and label encoder over model names "make/model".
    """
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    make_dirs = [d for d in glob.glob(os.path.join(paths.ds1, split, "*")) if os.path.isdir(d)]
    for make_dir in sorted(make_dirs):
        make = os.path.basename(make_dir)
        model_dirs = [d for d in glob.glob(os.path.join(make_dir, "*")) if os.path.isdir(d)]
        for model_dir in sorted(model_dirs):
            model = os.path.basename(model_dir)
            label = f"{make}/{model}"
            files = sorted(glob.glob(os.path.join(model_dir, "*.wav")))
            for fp in files:
                audio = load_audio(fp)
                base_audios = [audio]
                if augment:
                    base_audios.extend(frequency_warping(audio))
                for a in base_audios:
                    windows = preprocess_audio_array(a)
                    for w in windows:
                        X_list.append(w)
                        y_list.append(label)

    if not y_list:
        le = LabelEncoder().fit([])
        X = np.empty((0, AUDIO_CONFIG["time_steps"], AUDIO_CONFIG["n_mfcc"]))
        y = np.empty((0, 0))
        return X, y, le
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_list)
    y = to_categorical(y_encoded, num_classes=len(le.classes_))
    X = np.asarray(X_list, dtype=np.float32)
    return X, y, le
