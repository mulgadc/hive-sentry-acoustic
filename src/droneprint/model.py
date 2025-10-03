import tensorflow as tf
from tensorflow.keras import layers, models
from .config import AUDIO_CONFIG, TRAINING_CONFIG


def create_lstm_model(input_shape=None, num_classes: int = 2, learning_rate: float = None) -> tf.keras.Model:
    """Create LSTM model per spec.

    input_shape: (time_steps, n_mfcc). If None, derived from config.
    """
    if input_shape is None:
        input_shape = (AUDIO_CONFIG["time_steps"], AUDIO_CONFIG["n_mfcc"])  # (10, 40)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(TRAINING_CONFIG["hidden_units"], return_sequences=True, dropout=TRAINING_CONFIG["dropout_rate"]),
            layers.LSTM(TRAINING_CONFIG["hidden_units"], dropout=TRAINING_CONFIG["dropout_rate"]),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    lr = TRAINING_CONFIG["learning_rate"] if learning_rate is None else learning_rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
