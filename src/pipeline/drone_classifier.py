from __future__ import annotations

from typing import Optional, Dict, Any, Deque
from collections import deque
import os
import numpy as np
import librosa

from src.messages import AudioFrame, DOAResult, Classification
from src.config import SAMPLE_RATE
from src.config import MODELS_ROOT as CONFIG_MODELS_ROOT

try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional standalone Keras 3 (preferred for .h5 saved with Keras 3.x)
try:
    import keras  # type: ignore
    KERAS_AVAILABLE = True
except Exception:
    keras = None  # type: ignore
    KERAS_AVAILABLE = False

# Optional GPU DSP via PyTorch/torchaudio
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
try:
    import torchaudio  # type: ignore
    TORCHAUDIO_AVAILABLE = True
except Exception:
    torchaudio = None  # type: ignore
    TORCHAUDIO_AVAILABLE = False


class DroneClassifier:
    """Streaming classifier wrapping the TensorFlow models in models root.
    - X: binary drone detector (classifier_x_binary.h5)
    - Y: make classifier (classifier_y_make.h5 + classifier_y_make_labels.txt)
    - Z: optional per-make classifiers in classifier_z_<make>/ with labels.txt

    Maintains a rolling 200 ms sample buffer and rolling MFCC frames buffer (40-dim).
    Forms a (1, 10, 40) sequence when enough frames are available and runs the cascade.
    """

    def __init__(self, enabled: bool = True):
        # Enable only if requested AND TensorFlow is importable
        self.enabled = bool(enabled) and TF_AVAILABLE
        # Models root (env override)
        self._models_root = os.environ.get("MODELS_ROOT", CONFIG_MODELS_ROOT)
        # Thresholds (env override)
        try:
            self._thr_x = float(os.environ.get("CLS_X_THRESHOLD", "0.9"))
        except Exception:
            self._thr_x = 0.9
        try:
            self._thr_y = float(os.environ.get("CLS_Y_THRESHOLD", "0.6"))
        except Exception:
            self._thr_y = 0.6
        try:
            self._pred_period_s = float(os.environ.get("CLS_PRED_PERIOD_S", "2.0"))
        except Exception:
            self._pred_period_s = 2.0

        # Runtime buffers
        # Tunables (env overrides)
        try:
            self._sr_target = int(os.environ.get("CLS_SR_TARGET", "48000"))
        except Exception:
            self._sr_target = 48000
        self._win_s = 0.2  # 200 ms
        self._seq_steps = 10  # 10 frames per sequence
        # Accumulate samples only; compute MFCC at prediction time
        self._acc_samples: Deque[float] = deque(maxlen=int(self._sr_target * 3.0))  # ~3s at target SR
        self._mfcc_frames: Deque[np.ndarray] = deque(maxlen=self._seq_steps)
        self._samples_since_pred: int = 0
        self._last_sr: int = int(SAMPLE_RATE)

        # GPU DSP enable
        try:
            use_gpu_env = os.environ.get("CLS_USE_GPU", "1")
            self._use_gpu_dsp = (use_gpu_env not in ("0", "false", "False")) and TORCH_AVAILABLE and TORCHAUDIO_AVAILABLE and torch.cuda.is_available()
        except Exception:
            self._use_gpu_dsp = False
        self._mfcc_torch = None
        self._resample_last_from = None
        if self._use_gpu_dsp:
            try:
                # Build MFCC transform on GPU for target SR
                self._mfcc_torch = torchaudio.transforms.MFCC(
                    sample_rate=self._sr_target,
                    n_mfcc=40,
                    melkwargs={
                        "n_fft": 2048,
                        "hop_length": 512,
                        "f_max": 8000.0,
                        "center": True,
                    },
                ).to("cuda")
                self._resample_last_from = None
                print("[cls] GPU DSP enabled (PyTorch+torchaudio)")
            except Exception:
                self._use_gpu_dsp = False

        # Models
        self._m_x = None
        self._m_y = None
        self._m_z: Dict[str, Any] = {}  # key: make
        self._labels_y: list[str] = []
        self._labels_z: Dict[str, list[str]] = {}
        # Log TF availability and models root once
        try:
            print(f"[cls] TensorFlow available={TF_AVAILABLE} | MODELS_ROOT={self._models_root}")
        except Exception:
            pass
        if self.enabled:
            self._load_models()

    # --- Public API ---
    def preprocess(self, frame: AudioFrame, doa: Optional[DOAResult] = None):
        if not self.enabled:
            return None
        try:
            # Accumulate raw mono samples; resampling/MFCC is deferred to predict()
            mono = frame.audio[:, 0].astype(np.float32, copy=False)
            self._acc_samples.extend(mono.tolist())
            self._last_sr = int(frame.sr)
            # Advance logical counter in TARGET SR units to trigger predictions by time
            try:
                if int(frame.sr) > 0:
                    inc = int(round(len(mono) * (self._sr_target / float(frame.sr))))
                else:
                    inc = len(mono)
                self._samples_since_pred += inc
            except Exception:
                self._samples_since_pred += len(mono)
        except Exception:
            pass
        return None

    def predict(self, _unused) -> Optional[Classification]:
        if not self.enabled:
            return None
        try:
            # Throttle: only run prediction when at least pred_period_s of new samples accumulated
            if self._samples_since_pred < int(self._pred_period_s * self._sr_target):
                return None
            # Build MFCCs from the latest 200 ms window at prediction time
            target_len_src = int(round(self._win_s * max(1, self._last_sr)))
            if len(self._acc_samples) < target_len_src:
                return None
            y_src = np.array(list(self._acc_samples)[-target_len_src:], dtype=np.float32)
            # Resample to target SR if needed and compute MFCC
            if self._use_gpu_dsp:
                try:
                    y = torch.from_numpy(y_src).to(torch.float32).to("cuda")[None, :]
                    if int(self._last_sr) != int(self._sr_target):
                        y = torchaudio.functional.resample(y, orig_freq=int(self._last_sr), new_freq=int(self._sr_target))
                    # MFCC -> (B, n_mfcc, T)
                    mfcc_t = self._mfcc_torch(y)  # type: ignore[arg-type]
                    # Normalize per-frame (L2)
                    mfcc_t = mfcc_t.transpose(1, 2)  # (B, T, 40)
                    v = mfcc_t.squeeze(0).contiguous().float().cpu().numpy()
                except Exception:
                    # Fallback to CPU
                    y = y_src
                    if int(self._last_sr) != int(self._sr_target):
                        y = librosa.resample(y, orig_sr=int(self._last_sr), target_sr=int(self._sr_target))
                    v = self._compute_mfcc_cpu(y)
            else:
                y = y_src
                if int(self._last_sr) != int(self._sr_target):
                    y = librosa.resample(y, orig_sr=int(self._last_sr), target_sr=int(self._sr_target))
                v = self._compute_mfcc_cpu(y)
            # Update rolling MFCC frames and build 10x40 sequence
            for i in range(v.shape[0]):
                self._mfcc_frames.append(v[i].astype(np.float32))
            if len(self._mfcc_frames) < self._seq_steps:
                # Not enough history yet; keep accumulating
                return None
            seq = np.stack(list(self._mfcc_frames)[-self._seq_steps:], axis=0).astype(np.float32)
            # X
            p_drone, logits = self._predict_x(seq)
            if p_drone is None:
                return None
            if p_drone < self._thr_x:
                # Reset throttle counter even on Non-Drone decision
                self._samples_since_pred = 0
                return Classification(label="Non-Drone", prob=float(p_drone), logits=logits)
            # Y
            make_label = None
            make_probs = None
            if self._m_y is not None:
                make_probs = self._predict_probs(self._m_y, seq)
                if make_probs is not None and make_probs.size > 0:
                    mk_idx = int(np.argmax(make_probs))
                    if self._labels_y and mk_idx < len(self._labels_y):
                        make_label = self._labels_y[mk_idx]
                    elif make_probs is not None:
                        make_label = f"MAKE_{mk_idx}"
                    # Confidence gate for unknown make
                    top = float(make_probs[mk_idx])
                    if top < self._thr_y:
                        make_label = "Unknown"
            # Z
            model_label = None
            if make_label and make_label in self._m_z:
                probs_z = self._predict_probs(self._m_z[make_label], seq)
                if probs_z is not None and probs_z.size > 0:
                    md_idx = int(np.argmax(probs_z))
                    labels = self._labels_z.get(make_label, [])
                    if labels and md_idx < len(labels):
                        model_label = labels[md_idx]
                    else:
                        model_label = f"MODEL_{md_idx}"

            # Build label string
            if make_label and model_label:
                label = f"Drone: {make_label}/{model_label}"
            elif make_label:
                label = f"Drone: {make_label}"
            else:
                label = "Drone"
            # Reset throttle counter after a prediction
            self._samples_since_pred = 0
            return Classification(label=label, prob=float(p_drone), logits=logits)
        except Exception:
            return None

    # --- Internals ---
    def _load_models(self):
        any_loaded = False
        try:
            x_path = os.path.join(self._models_root, "classifier_x_binary.h5")
            y_path = os.path.join(self._models_root, "classifier_y_make.h5")
            # X model (binary drone detector)
            if os.path.exists(x_path):
                try:
                    if KERAS_AVAILABLE:
                        self._m_x = keras.models.load_model(x_path, compile=False)
                    elif TF_AVAILABLE:
                        self._m_x = tf.keras.models.load_model(x_path, compile=False)
                    else:
                        raise RuntimeError("Neither keras nor tf.keras available to load models")
                    any_loaded = True
                    print(f"[cls] Loaded X model: {x_path}")
                except Exception as e:
                    print(f"[cls][warn] Failed to load X model {x_path}: {e}")
            else:
                print(f"[cls][warn] X model not found at {x_path}")
            # Y model (make classifier)
            if os.path.exists(y_path):
                try:
                    if KERAS_AVAILABLE:
                        self._m_y = keras.models.load_model(y_path, compile=False)
                    elif TF_AVAILABLE:
                        self._m_y = tf.keras.models.load_model(y_path, compile=False)
                    else:
                        raise RuntimeError("Neither keras nor tf.keras available to load models")
                    any_loaded = True
                    print(f"[cls] Loaded Y model: {y_path}")
                except Exception as e:
                    print(f"[cls][warn] Failed to load Y model {y_path}: {e}")
                lblp = os.path.join(self._models_root, "classifier_y_make_labels.txt")
                if os.path.exists(lblp):
                    try:
                        with open(lblp, "r", encoding="utf-8") as f:
                            self._labels_y = [ln.strip() for ln in f if ln.strip()]
                        print(f"[cls] Loaded Y labels: {len(self._labels_y)} entries")
                    except Exception as e:
                        print(f"[cls][warn] Failed to load Y labels {lblp}: {e}")
                else:
                    print(f"[cls][warn] Y labels not found at {lblp}")
            else:
                print(f"[cls][warn] Y model not found at {y_path}")
            # Z models (optional per-make)
            try:
                for entry in os.listdir(self._models_root):
                    if entry.startswith("classifier_z_"):
                        make = entry.replace("classifier_z_", "")
                        d = os.path.join(self._models_root, entry)
                        mp = os.path.join(d, "model.h5")
                        lp = os.path.join(d, "labels.txt")
                        if os.path.exists(mp):
                            try:
                                if KERAS_AVAILABLE:
                                    self._m_z[make] = keras.models.load_model(mp, compile=False)
                                elif TF_AVAILABLE:
                                    self._m_z[make] = tf.keras.models.load_model(mp, compile=False)
                                else:
                                    raise RuntimeError("Neither keras nor tf.keras available to load models")
                                any_loaded = True
                                print(f"[cls] Loaded Z model for make '{make}': {mp}")
                            except Exception as e:
                                print(f"[cls][warn] Failed to load Z model for make '{make}' at {mp}: {e}")
                        if os.path.exists(lp):
                            try:
                                with open(lp, "r", encoding="utf-8") as f:
                                    self._labels_z[make] = [ln.strip() for ln in f if ln.strip()]
                                print(f"[cls] Loaded Z labels for '{make}': {len(self._labels_z[make])} entries")
                            except Exception as e:
                                print(f"[cls][warn] Failed to load Z labels for '{make}' at {lp}: {e}")
            except Exception:
                # directory listing failure shouldn't kill classifier
                pass
        except Exception as e:
            print(f"[cls][warn] Unexpected error while scanning models: {e}")
        # Only disable if nothing loaded at all
        if not any_loaded:
            self.enabled = False
            print("[cls] No models loaded; classifier disabled")
        else:
            print("[cls] Classifier enabled")

    def _compute_mfcc_cpu(self, y: np.ndarray) -> np.ndarray:
        # Peak normalize per 200 ms window
        try:
            mx = float(np.max(np.abs(y)))
            if mx > 0:
                y = (y / mx).astype(np.float32)
            # MFCC settings aligned with training (center=True)
            try:
                n_fft = int(os.environ.get("CLS_MFCC_NFFT", "2048"))
            except Exception:
                n_fft = 2048
            try:
                hop = int(os.environ.get("CLS_MFCC_HOP", "512"))
            except Exception:
                hop = 512
            mfcc = librosa.feature.mfcc(y=y, sr=self._sr_target, n_mfcc=40, fmax=8000.0, n_fft=n_fft, hop_length=hop, center=True)
            v = mfcc.T.astype(np.float32) if mfcc.ndim > 1 else mfcc.astype(np.float32).reshape(1, -1)
            # L2 rescale rows
            for i in range(v.shape[0]):
                nrm = float(np.linalg.norm(v[i]))
                if nrm > 0:
                    v[i] = (v[i] / nrm).astype(np.float32)
            return v
        except Exception:
            return np.zeros((1, 40), dtype=np.float32)

    def _predict_probs(self, model, seq10x40: np.ndarray) -> Optional[np.ndarray]:
        try:
            x = seq10x40.astype(np.float32)
            # Try common shapes: (1,10,40) and (1,10,40,1)
            try:
                p = model.predict(x[None, ...], verbose=0)
                return np.array(p).squeeze()
            except Exception:
                p = model.predict(x[None, ..., None], verbose=0)
                return np.array(p).squeeze()
        except Exception:
            return None

    def _predict_x(self, seq10x40: np.ndarray) -> tuple[Optional[float], Optional[list[float]]]:
        if self._m_x is None:
            return None, None
        probs = self._predict_probs(self._m_x, seq10x40)
        if probs is None:
            return None, None
        # Map to probability of class index 1 (Drone) for binary softmax or sigmoid
        if np.ndim(probs) == 0:
            p_drone = float(probs)
            logits = [1.0 - p_drone, p_drone]
        elif np.size(probs) == 1:
            p_drone = float(probs)
            logits = [1.0 - p_drone, p_drone]
        elif len(probs) == 2:
            p_drone = float(probs[1])
            logits = [float(probs[0]), float(probs[1])]
        else:
            p_drone = float(np.max(probs))
            logits = list(map(float, np.ravel(probs)))
        return p_drone, logits
