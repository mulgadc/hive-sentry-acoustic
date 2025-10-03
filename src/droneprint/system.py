import os
import json
from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa

from .config import AUDIO_CONFIG
from .preprocessing import preprocess_audio_file, preprocess_audio_array


class DronePrintSystem:
    def __init__(self):
        self.classifier_x: tf.keras.Model = None  # Drone detection (binary)
        self.classifier_y: tf.keras.Model = None  # Drone make identification
        self.classifiers_z: Dict[str, tf.keras.Model] = {}  # Drone model identification per make
        self.make_labels: list = []
        self.model_labels_per_make: Dict[str, list] = {}
        # OpenMax-style closed-set detector
        self.openmax_closedset: tf.keras.Model = None
        self.openmax_labels: List[str] = []  # labels like "make/model"
        self.openmax_thresholds: Dict[str, float] = {}  # class-wise thresholds

    def load_from_dir(self, models_dir: str):
        """Load saved models and label files from a directory."""
        # Binary
        x_path = os.path.join(models_dir, "classifier_x_binary.h5")
        if os.path.isfile(x_path):
            self.classifier_x = tf.keras.models.load_model(x_path)

        # Make Y
        y_path = os.path.join(models_dir, "classifier_y_make.h5")
        if os.path.isfile(y_path):
            self.classifier_y = tf.keras.models.load_model(y_path)
            labels_txt = os.path.join(models_dir, "classifier_y_make_labels.txt")
            if os.path.isfile(labels_txt):
                with open(labels_txt, "r") as f:
                    self.make_labels = [line.strip() for line in f if line.strip()]

        # Z per make
        for entry in os.listdir(models_dir):
            if entry.startswith("classifier_z_"):
                make = entry[len("classifier_z_"):]
                make_dir = os.path.join(models_dir, entry)
                model_path = os.path.join(make_dir, "model.h5")
                labels_path = os.path.join(make_dir, "labels.txt")
                if os.path.isfile(model_path):
                    self.classifiers_z[make] = tf.keras.models.load_model(model_path)
                    if os.path.isfile(labels_path):
                        with open(labels_path, "r") as f:
                            self.model_labels_per_make[make] = [line.strip() for line in f if line.strip()]

        # OpenMax-style closed-set model
        om_model = os.path.join(models_dir, "openmax_closedset.h5")
        om_labels = os.path.join(models_dir, "openmax_labels.txt")
        om_thresh = os.path.join(models_dir, "openmax_thresholds.json")
        if os.path.isfile(om_model):
            self.openmax_closedset = tf.keras.models.load_model(om_model)
            if os.path.isfile(om_labels):
                with open(om_labels, "r") as f:
                    self.openmax_labels = [line.strip() for line in f if line.strip()]
            if os.path.isfile(om_thresh):
                with open(om_thresh, "r") as f:
                    self.openmax_thresholds = json.load(f)

    def _average_over_6s_windows(self, probs: np.ndarray) -> np.ndarray:
        """Average predictions over three consecutive 6-second windows.
        probs: (N, C) per 2s sequence (200 ms step size assumed with stride 1 seq)
        6 s ~ 30 frames => ~15 sequences (since each seq advances by 1 frame of 200 ms)
        We aggregate 15 consecutive seqs per 6 s window. Then average 3 such windows (0, +15, +30).
        If insufficient length, fall back to simple mean over all.
        Returns averaged (C,) vector.
        """
        seqs_per_6s = 15
        starts = [0, seqs_per_6s, 2 * seqs_per_6s]
        means = []
        for s in starts:
            e = s + seqs_per_6s
            if e <= len(probs):
                means.append(np.mean(probs[s:e], axis=0))
        if means:
            return np.mean(np.stack(means, axis=0), axis=0)
        else:
            return np.mean(probs, axis=0)

    def _cascade_predict(self, features: np.ndarray, detector_mode: str = "binary") -> str:
        """Run the cascade on precomputed features and return the prediction string."""
        if features.size == 0:
            return "No audio content"

        # Step 1: Drone detection
        if detector_mode == "openmax":
            if self.openmax_closedset is None or not self.openmax_labels or not self.openmax_thresholds:
                return "OpenMax detector not loaded"
            cs_probs = self.openmax_closedset.predict(features, verbose=0)  # (N, K)
            avg_probs = self._average_over_6s_windows(cs_probs)  # (K,)
            # Determine top class and apply class-wise threshold
            top_idx = int(np.argmax(avg_probs))
            top_label = self.openmax_labels[top_idx] if self.openmax_labels else str(top_idx)
            top_prob = float(avg_probs[top_idx])
            thr = float(self.openmax_thresholds.get(top_label, 0.5))
            if top_prob < thr:
                return "Non-Drone"
            # Continue cascade using Y/Z if available, else return make/model inferred from top_label
            parts = top_label.split("/", 1)
            if len(parts) == 2:
                predicted_make, predicted_model = parts
                if predicted_make in self.classifiers_z:
                    # refine with per-make model classifier if available
                    model_probs = self.classifiers_z[predicted_make].predict(features, verbose=0)
                    model_avg = self._average_over_6s_windows(model_probs)
                    labels = self.model_labels_per_make.get(predicted_make, [])
                    m_idx = int(np.argmax(model_avg))
                    predicted_model = labels[m_idx] if labels and m_idx < len(labels) else str(m_idx)
                return f"{predicted_make}: {predicted_model}"
            else:
                return top_label
        else:
            # Binary Classifier X
            if self.classifier_x is None:
                return "Classifier X not loaded"
            drone_probs = self.classifier_x.predict(features, verbose=0)
            avg_drone_probs = self._average_over_6s_windows(drone_probs)
            # Probability of class 1 (drone)
            p_drone = avg_drone_probs[1] if drone_probs.shape[1] > 1 else avg_drone_probs[0]
            if p_drone < 0.5:
                return "Non-Drone"

        # Step 2: Drone make identification (Classifier Y)
        if self.classifier_y is None:
            return "Drone"
        make_probs = self.classifier_y.predict(features, verbose=0)
        make_avg = self._average_over_6s_windows(make_probs)
        make_idx = int(np.argmax(make_avg))
        predicted_make = self.make_labels[make_idx] if self.make_labels and make_idx < len(self.make_labels) else str(make_idx)

        # Step 3: Drone model identification (Classifier Z)
        if predicted_make in self.classifiers_z:
            model_probs = self.classifiers_z[predicted_make].predict(features, verbose=0)
            model_avg = self._average_over_6s_windows(model_probs)
            model_idx = int(np.argmax(model_avg))
            labels = self.model_labels_per_make.get(predicted_make, [])
            predicted_model = labels[model_idx] if labels and model_idx < len(labels) else str(model_idx)
            return f"{predicted_make}: {predicted_model}"

        return predicted_make

    def predict(self, audio_file: str, detector_mode: str = "binary") -> str:
        """Complete prediction pipeline on a file (mono mix)."""
        features = preprocess_audio_file(audio_file)
        return self._cascade_predict(features, detector_mode)

    def predict_single_channel(self, audio_file: str, channel: int, detector_mode: str = "binary") -> str:
        """Predict on a specific channel from a multi-channel WAV without mixing.
        channel: 0-based channel index.
        """
        # Load with soundfile to preserve channels
        data, sr = sf.read(audio_file, always_2d=True)
        if data.ndim != 2 or data.shape[1] <= channel:
            return f"Channel {channel} not available (file has {data.shape[1] if data.ndim==2 else 1} channels)"
        y = data[:, channel].astype(np.float32, copy=False)
        target_sr = AUDIO_CONFIG["sample_rate"]
        if int(sr) != int(target_sr):
            try:
                y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
            except Exception:
                return f"Resample failed from {sr} to {target_sr}"
        features = preprocess_audio_array(y, sr=int(target_sr))
        return self._cascade_predict(features, detector_mode)
