import os
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf

from .config import TRAINING_CONFIG, AUDIO_CONFIG
from .model import create_lstm_model
from .data import (
    discover_dataset_paths,
    build_binary_dataset,
    build_make_dataset,
    build_model_datasets_per_make,
    build_all_drone_models_dataset,
)


def train_binary_classifier(
    datasets_root: str,
    output_dir: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    augment_splits: tuple = ("training",),
    noise_mix_pct_train: float = 0.0,
    noise_mix_pct_val: float = 0.0,
) -> tf.keras.Model:
    paths = discover_dataset_paths(datasets_root)
    X_train, y_train = build_binary_dataset(
        paths,
        split="training",
        augment=("training" in augment_splits),
        noise_mix_pct=noise_mix_pct_train,
    )
    X_val, y_val = build_binary_dataset(
        paths,
        split="validation",
        augment=("validation" in augment_splits),
        noise_mix_pct=noise_mix_pct_val,
    )

    model = create_lstm_model(num_classes=2, learning_rate=learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=TRAINING_CONFIG["epochs"] if epochs is None else epochs,
        batch_size=TRAINING_CONFIG["batch_size"] if batch_size is None else batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "classifier_x_binary.h5"))
    return model


def train_make_classifier(
    datasets_root: str,
    output_dir: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    augment_splits: tuple = ("training",),
) -> Tuple[tf.keras.Model, Dict]:
    paths = discover_dataset_paths(datasets_root)
    X_train, y_train, le_make = build_make_dataset(
        paths,
        split="training",
        augment=("training" in augment_splits),
    )
    X_val, y_val, _ = build_make_dataset(
        paths,
        split="validation",
        augment=("validation" in augment_splits),
    )

    num_classes = y_train.shape[1] if y_train.size else 0
    if num_classes == 0:
        raise RuntimeError("No classes found for make classifier. Check dataset structure.")

    model = create_lstm_model(num_classes=num_classes, learning_rate=learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    ]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=TRAINING_CONFIG["epochs"] if epochs is None else epochs,
        batch_size=TRAINING_CONFIG["batch_size"] if batch_size is None else batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "classifier_y_make.h5"))
    # Save label mapping
    with open(os.path.join(output_dir, "classifier_y_make_labels.txt"), "w") as f:
        for cls in le_make.classes_:
            f.write(f"{cls}\n")

    return model, {"label_encoder_classes": le_make.classes_.tolist()}


def train_model_classifiers(
    datasets_root: str,
    output_dir: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    augment_splits: tuple = ("training",),
) -> Dict[str, Tuple[tf.keras.Model, Dict]]:
    paths = discover_dataset_paths(datasets_root)
    datasets = build_model_datasets_per_make(
        paths,
        split="training",
        augment=("training" in augment_splits),
    )
    val_datasets = build_model_datasets_per_make(
        paths,
        split="validation",
        augment=("validation" in augment_splits),
    )

    results: Dict[str, Tuple[tf.keras.Model, Dict]] = {}
    os.makedirs(output_dir, exist_ok=True)

    for make, (X_train, y_train, le_model) in datasets.items():
        X_val, y_val, _ = val_datasets.get(make, (np.empty((0, AUDIO_CONFIG["time_steps"], AUDIO_CONFIG["n_mfcc"])), np.empty((0, 0)), None))
        num_classes = y_train.shape[1] if y_train.size else 0
        if num_classes <= 1:
            # Not enough models to train per-make classifier
            continue
        model = create_lstm_model(num_classes=num_classes, learning_rate=learning_rate)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)]
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=TRAINING_CONFIG["epochs"] if epochs is None else epochs,
            batch_size=TRAINING_CONFIG["batch_size"] if batch_size is None else batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        make_dir = os.path.join(output_dir, f"classifier_z_{make}")
        os.makedirs(make_dir, exist_ok=True)
        model.save(os.path.join(make_dir, f"model.h5"))
        with open(os.path.join(make_dir, "labels.txt"), "w") as f:
            for cls in le_model.classes_:
                f.write(f"{cls}\n")
        results[make] = (model, {"label_encoder_classes": le_model.classes_.tolist()})

    return results


def train_openmax_closedset(
    datasets_root: str,
    output_dir: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    augment_splits: tuple = ("training",),
) -> Tuple[tf.keras.Model, Dict]:
    """Train a closed-set multi-class model over all DS1 models and compute class-wise thresholds
    using validation distributions of top predicted probability.
    Saves:
      - openmax_closedset.h5
      - openmax_labels.txt
      - openmax_thresholds.json
    """
    import json

    paths = discover_dataset_paths(datasets_root)
    X_train, y_train, le = build_all_drone_models_dataset(paths, split="training", augment=("training" in augment_splits))
    X_val, y_val, _ = build_all_drone_models_dataset(paths, split="validation", augment=("validation" in augment_splits))

    num_classes = y_train.shape[1] if y_train.size else 0
    if num_classes == 0:
        raise RuntimeError("No classes found for OpenMax closed-set model. Check DS1 structure.")

    model = create_lstm_model(num_classes=num_classes, learning_rate=learning_rate)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)]
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=TRAINING_CONFIG["epochs"] if epochs is None else epochs,
        batch_size=TRAINING_CONFIG["batch_size"] if batch_size is None else batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Compute class-wise thresholds from validation set: take 5th percentile of top-prob for each true class
    val_probs = model.predict(X_val, verbose=0)
    y_true = np.argmax(y_val, axis=1)
    thresholds: Dict[str, float] = {}
    labels = le.classes_.tolist()
    for cls_idx, label in enumerate(labels):
        # top prob predicted for the true class
        cls_mask = y_true == cls_idx
        if not np.any(cls_mask):
            thresholds[label] = 0.5
            continue
        probs_for_cls = val_probs[cls_mask, cls_idx]
        thr = float(np.percentile(probs_for_cls, 5))  # 5th percentile
        # Clamp to sensible minimum
        thresholds[label] = max(0.3, thr)

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "openmax_closedset.h5"))
    with open(os.path.join(output_dir, "openmax_labels.txt"), "w") as f:
        for lbl in labels:
            f.write(f"{lbl}\n")
    with open(os.path.join(output_dir, "openmax_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    return model, {"labels": labels, "thresholds": thresholds}
