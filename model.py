import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_model(
    sequence_length=20,
    feature_dim=63,
    num_classes=10,
    learning_rate=1e-3,
    decay_steps=None,
):
    """Build a stacked LSTM model for sign language sequence classification."""
    inputs = layers.Input(shape=(sequence_length, feature_dim))
    x = layers.GaussianNoise(0.015)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Bidirectional(layers.LSTM(192, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(128, dropout=0.2))(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    learning_rate_schedule = (
        tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=0.05,
        )
        if decay_steps
        else learning_rate
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )

    return model
