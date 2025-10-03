AUDIO_CONFIG = {
    "sample_rate": 48000,
    "frame_duration_ms": 200,
    "frame_overlap": 0,
    "fmax": 8000,
    "n_mfcc": 40,
    "time_steps": 10,
}

TRAINING_CONFIG = {
    "epochs": 500,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "hidden_units": 32,
    "dropout_rate": 0.2,
    "validation_split": 0.2,
}

DATA_SPLITS = {
    "training": 0.6,
    "validation": 0.2,
    "testing": 0.2,
}
