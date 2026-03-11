"""
Model architecture registry for depression detection.

All model architectures live here. New architectures can be added
by defining a builder function and registering it in MODEL_REGISTRY.
This follows the Open/Closed principle: extend without modifying
existing code.

Each builder returns a compiled-ready Keras model (not compiled).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Architecture builders ─────────────────────────────────────────────

def create_mel_cnn(input_shape=(128, 313, 1), dropout_rate=0.3) -> keras.Model:
    """
    4-block CNN for mel spectrogram features.
    Uses BatchNorm + GlobalAvgPool for efficient mobile deployment.
    """
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1: (128, 313, 1) -> (64, 156, 32)
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate),
            # Block 2: (64, 156, 32) -> (32, 78, 64)
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate + 0.1),
            # Block 3: (32, 78, 64) -> (16, 39, 128)
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(dropout_rate + 0.1),
            # Block 4: global avg pool -> 128
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            # Classifier
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate + 0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="mel_cnn",
    )


def create_bilstm(input_shape=(313, 128), dropout_rate=0.3) -> keras.Model:
    """
    Bidirectional LSTM for sequential mel spectrogram analysis.
    Input: (time_steps, n_mels) — each time frame is a 128-dim vector.
    """
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=dropout_rate)
            ),
            layers.BatchNormalization(),
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False, dropout=dropout_rate)
            ),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout_rate + 0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="bilstm",
    )


def create_cnn_lstm(input_shape=(128, 313, 1), dropout_rate=0.3) -> keras.Model:
    """
    CNN-LSTM Hybrid: 2 CNN blocks extract local features,
    then LSTM models temporal dependencies.
    """
    inputs = layers.Input(shape=input_shape)

    # CNN Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # CNN Block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Reshape to sequence: (freq/4, time/4, 64) -> (time/4, freq/4 * 64)
    freq_bins = input_shape[0] // 4
    time_steps = input_shape[1] // 4
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((time_steps, freq_bins * 64))(x)

    # LSTM
    x = layers.LSTM(64, return_sequences=False, dropout=dropout_rate)(x)
    x = layers.BatchNormalization()(x)

    # Classifier
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_rate + 0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="cnn_lstm")


def create_cnn_attention(input_shape=(128, 313, 1), dropout_rate=0.3) -> keras.Model:
    """
    CNN + Multi-Head Self-Attention.
    2 CNN blocks reduce dimensions, then attention captures global temporal patterns.
    """
    inputs = layers.Input(shape=input_shape)

    # CNN Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # CNN Block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Reshape to sequence
    freq_bins = input_shape[0] // 4
    time_steps = input_shape[1] // 4
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((time_steps, freq_bins * 64))(x)

    # Project down for attention efficiency
    x = layers.Dense(128)(x)

    # Multi-Head Self-Attention with residual
    attn_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=dropout_rate
    )(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    # Global pooling over time
    x = layers.GlobalAveragePooling1D()(x)

    # Classifier
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate + 0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="cnn_attention")


def create_separable_cnn(input_shape=(128, 313, 1), dropout_rate=0.3) -> keras.Model:
    """
    MobileNet-style depthwise separable CNN.
    ~60% fewer parameters than standard CNN.
    """
    inputs = layers.Input(shape=input_shape)

    # Initial standard Conv to project from 1 channel
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    # Separable Block 1
    x = layers.SeparableConv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Separable Block 2
    x = layers.SeparableConv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Separable Block 3
    x = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)

    # Separable Block 4
    x = layers.SeparableConv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate + 0.1)(x)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate + 0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="separable_cnn")


def create_multi_feature_cnn(
    mel_shape=(128, 313, 1),
    mfcc_shape=(13, 313, 1),
    dropout_rate=0.3,
) -> keras.Model:
    """
    Dual-branch CNN fusing mel spectrogram and MFCC features.
    Branch A (Mel): 3 Conv2D blocks -> GlobalAvgPool -> 64-dim
    Branch B (MFCC): 2 Conv2D blocks -> GlobalAvgPool -> 64-dim
    Fusion: Concatenate -> Dense(64) -> Dense(1)
    """
    # Branch A: Mel Spectrogram
    mel_input = layers.Input(shape=mel_shape, name="mel_input")
    a = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(mel_input)
    a = layers.BatchNormalization()(a)
    a = layers.MaxPooling2D((2, 2))(a)
    a = layers.Dropout(dropout_rate)(a)

    a = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(a)
    a = layers.BatchNormalization()(a)
    a = layers.MaxPooling2D((2, 2))(a)
    a = layers.Dropout(dropout_rate)(a)

    a = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(a)
    a = layers.BatchNormalization()(a)
    a = layers.GlobalAveragePooling2D()(a)

    # Branch B: MFCC
    mfcc_input = layers.Input(shape=mfcc_shape, name="mfcc_input")
    b = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(mfcc_input)
    b = layers.BatchNormalization()(b)
    b = layers.MaxPooling2D((2, 2))(b)
    b = layers.Dropout(dropout_rate)(b)

    b = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(b)
    b = layers.BatchNormalization()(b)
    b = layers.GlobalAveragePooling2D()(b)

    # Fusion
    fused = layers.Concatenate()([a, b])
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dropout(dropout_rate + 0.2)(fused)
    output = layers.Dense(1, activation="sigmoid")(fused)

    return keras.Model(inputs=[mel_input, mfcc_input], outputs=output, name="multi_feature_cnn")


# ── Model registry ────────────────────────────────────────────────────
# Maps architecture name -> (builder_function, input_type, combined_builder_name)
# input_type: "cnn" | "lstm" | "dual"
# combined_builder: which build_combined_* function to use for TFLite export

MODEL_REGISTRY = {
    "mel_cnn": {
        "builder": create_mel_cnn,
        "input_type": "cnn",
        "combined_builder": "cnn",
        "tflite_name": "mel_cnn/mel_cnn_combined.tflite",
        "description": "4-block CNN on mel spectrograms",
    },
    "bilstm": {
        "builder": create_bilstm,
        "input_type": "lstm",
        "combined_builder": "lstm",
        "tflite_name": "bilstm/bilstm_combined.tflite",
        "description": "Bidirectional LSTM on mel spectrograms",
    },
    "cnn_lstm": {
        "builder": create_cnn_lstm,
        "input_type": "cnn",
        "combined_builder": "cnn",
        "tflite_name": "cnn_lstm/cnn_lstm_combined.tflite",
        "description": "CNN-LSTM hybrid architecture",
    },
    "cnn_attention": {
        "builder": create_cnn_attention,
        "input_type": "cnn",
        "combined_builder": "cnn",
        "tflite_name": "cnn_attention/cnn_attention_combined.tflite",
        "description": "CNN + Multi-Head Self-Attention",
    },
    "separable_cnn": {
        "builder": create_separable_cnn,
        "input_type": "cnn",
        "combined_builder": "cnn",
        "tflite_name": "separable_cnn/separable_cnn_combined.tflite",
        "description": "MobileNet-style depthwise separable CNN",
    },
    "multi_feature": {
        "builder": create_multi_feature_cnn,
        "input_type": "dual",
        "combined_builder": "dual_branch",
        "tflite_name": "multi_feature/multi_feature_combined.tflite",
        "description": "Dual-branch CNN (mel + MFCC fusion)",
    },
}


def get_model(name: str, **kwargs) -> keras.Model:
    """
    Factory function to get a model by name.

    Args:
        name: Architecture name (key in MODEL_REGISTRY).
        **kwargs: Passed to the builder (e.g. dropout_rate, input_shape).

    Raises:
        ValueError: If architecture name is not registered.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown architecture '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]["builder"](**kwargs)


def list_architectures() -> list:
    """Return list of available architecture names."""
    return list(MODEL_REGISTRY.keys())
