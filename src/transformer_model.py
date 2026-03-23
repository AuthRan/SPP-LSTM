"""
Transformer Model Module - Time-series Transformer for stock price prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Input, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

def positional_encoding(sequence_length: int, d_model: int) -> tf.Tensor:
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    Returns:
        Tensor of shape (1, sequence_length, d_model)
    """
    positions = np.arange(sequence_length)[:, np.newaxis]          # (T, 1)
    dims = np.arange(d_model)[np.newaxis, :]                       # (1, D)

    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)

    # Apply sin to even indices, cos to odd indices
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)    # (1, T, D)


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Single Transformer encoder block: MHA → Add & Norm → FFN → Add & Norm."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                       dropout=dropout_rate)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-Head Self-Attention
        attn_out = self.attn(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        # Feed-Forward Network
        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)
        return x

    def get_config(self):
        config = super().get_config()
        return config


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_transformer_model(
    sequence_length: int = 60,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 2,
    dropout_rate: float = 0.1,
    forecast_horizon: int = 1
) -> tf.keras.Model:
    """
    Build a Transformer encoder model for stock price prediction.

    Architecture:
        Input (sequence_length, 1)
        → Linear projection to d_model
        + Positional Encoding
        → N × TransformerEncoderBlock
        → GlobalAveragePooling1D
        → Dense(64, relu) → Dropout
        → Dense(forecast_horizon)

    Args:
        sequence_length : lookback window length (time steps)
        d_model         : internal embedding dimension
        num_heads       : number of attention heads (must divide d_model)
        ff_dim          : feed-forward hidden dimension
        num_layers      : number of stacked encoder blocks
        dropout_rate    : dropout probability
        forecast_horizon: output size (1 for single-step prediction)

    Returns:
        Compiled Keras functional model
    """
    inputs = Input(shape=(sequence_length, 1))

    # Project to d_model
    x = Dense(d_model)(inputs)

    # Add positional encoding (non-trainable)
    pos_enc = positional_encoding(sequence_length, d_model)
    x = x + pos_enc

    # Stack encoder blocks
    for _ in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout_rate)(x)

    # Aggregate sequence → vector
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(forecast_horizon if forecast_horizon > 1 else 1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae", "mse"])
    return model


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_transformer_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    model_save_path: str = "models",
    ticker: str = "DEFAULT"
) -> tuple:
    """
    Train the Transformer model with early stopping and checkpointing.

    Returns:
        Tuple of (trained_model, training_history, metrics)
    """
    save_dir = Path(model_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = ticker.replace(".", "_")   # Sanitize for Windows filenames
    checkpoint_path = str(save_dir / f"{safe_ticker}_transformer_{timestamp}.keras")
    best_model_path = str(save_dir / f"{safe_ticker}_transformer_best.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0          # Must be 0 under Streamlit on Windows (stdout flush issue)
    )

    model.save(str(best_model_path))

    # Metrics on scaled data
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)

    metrics = {
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    return model, history, metrics
