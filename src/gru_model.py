"""
GRU Model Module - GRU-based stock price prediction for comparison with LSTM
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from datetime import datetime


def build_gru_model(
    sequence_length: int = 60,
    units: int = 50,
    dropout_rate: float = 0.2,
    forecast_horizon: int = 1
) -> tf.keras.Model:
    """
    Build GRU (Gated Recurrent Unit) model for stock price prediction.

    GRU is similar to LSTM but has fewer parameters, making it faster to train.
    Architecture:
        - Input layer
        - GRU(50) with dropout
        - GRU(50) with dropout
        - Dense(25)
        - Dense(1) - Output

    Args:
        sequence_length: Length of input sequences
        units: Number of GRU units
        dropout_rate: Dropout rate for regularization
        forecast_horizon: Number of days to predict

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First GRU layer
        GRU(units, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(dropout_rate),

        # Second GRU layer
        GRU(units),
        Dropout(dropout_rate),

        # Dense layers
        Dense(25, activation='relu'),
        Dense(forecast_horizon if forecast_horizon > 1 else 1)
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae', 'mse']
    )

    return model


def train_gru_model(
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
    Train the GRU model with early stopping and model checkpointing.

    Args:
        model: Compiled Keras model
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data
        y_test: Test target data
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        model_save_path: Directory to save trained models
        ticker: Stock ticker for naming the model file

    Returns:
        Tuple of (trained_model, training_history, metrics)
    """
    # Create save directory
    save_dir = Path(model_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = str(save_dir / f"{ticker}_gru_{timestamp}.keras")
    best_model_path = str(save_dir / f"{ticker}_gru_best.keras")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Save the best model
    model.save(str(best_model_path))

    # Calculate metrics
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))

    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)

    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1]
    }

    return model, history, metrics


def load_gru_model(model_path: str) -> tf.keras.Model:
    """
    Load a pre-trained GRU model.

    Args:
        model_path: Path to the saved model (.keras file)

    Returns:
        Loaded Keras model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    return load_model(model_path)


def get_gru_model_path(ticker: str, model_dir: str = "models") -> str:
    """
    Get the path to the best saved GRU model for a ticker.

    Args:
        ticker: Stock ticker symbol
        model_dir: Models directory

    Returns:
        Path to the best model file
    """
    model_dir = Path(model_dir)
    pattern = f"{ticker}_gru_best.keras"
    model_path = model_dir / pattern

    if model_path.exists():
        return str(model_path)

    # Fallback: find any GRU model for this ticker
    models = list(model_dir.glob(f"{ticker}_gru_*.keras"))
    if models:
        return str(sorted(models)[-1])

    return None
