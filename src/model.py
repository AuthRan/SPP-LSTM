"""
LSTM Model Module - Build, train, and save LSTM models for stock prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import os
from datetime import datetime


def build_lstm_model(
    sequence_length: int = 60,
    units: int = 50,
    dropout_rate: float = 0.2,
    forecast_horizon: int = 1
) -> tf.keras.Model:
    """
    Build LSTM model for stock price prediction.

    Architecture:
        - Input layer
        - LSTM(50) with dropout
        - LSTM(50) with dropout
        - Dense(25)
        - Dense(1) - Output

    Args:
        sequence_length: Length of input sequences
        units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        forecast_horizon: Number of days to predict (for multi-output)

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(units, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(dropout_rate),

        # Second LSTM layer
        LSTM(units),
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


def build_enhanced_lstm_model(
    sequence_length: int = 60,
    units: list = None,
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """
    Build an enhanced LSTM model with configurable layers.

    Args:
        sequence_length: Length of input sequences
        units: List of units for each LSTM layer (default: [100, 50, 50])
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    if units is None:
        units = [100, 50, 50]

    model = Sequential()

    # First LSTM layer with return_sequences=True
    model.add(LSTM(units[0], return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(dropout_rate))

    # Middle LSTM layers
    for i, unit_count in enumerate(units[1:-1]):
        model.add(LSTM(unit_count, return_sequences=True))
        model.add(Dropout(dropout_rate))

    # Last LSTM layer
    model.add(LSTM(units[-1]))
    model.add(Dropout(dropout_rate))

    # Dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='huber',  # Huber loss is robust to outliers
        metrics=['mae', 'mse']
    )

    return model


def train_model(
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
    Train the LSTM model with early stopping and model checkpointing.

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
    checkpoint_path = save_dir / f"{ticker}_lstm_{timestamp}.keras"
    best_model_path = save_dir / f"{ticker}_lstm_best.keras"

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

    # Save the best model with a standard name
    model.save(best_model_path)

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


def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load a pre-trained LSTM model.

    Args:
        model_path: Path to the saved model (.keras file)

    Returns:
        Loaded Keras model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    return load_model(model_path)


def get_default_model_path(ticker: str, model_dir: str = "models") -> str:
    """
    Get the path to the best saved model for a ticker.

    Args:
        ticker: Stock ticker symbol
        model_dir: Models directory

    Returns:
        Path to the best model file
    """
    model_dir = Path(model_dir)
    pattern = f"{ticker}_lstm_best.keras"
    model_path = model_dir / pattern

    if model_path.exists():
        return str(model_path)

    # Fallback: find any model for this ticker
    models = list(model_dir.glob(f"{ticker}_lstm_*.keras"))
    if models:
        return str(sorted(models)[-1])

    return None


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler
) -> dict:
    """
    Evaluate model performance with various metrics.

    Args:
        model: Trained model
        X_test: Test input data
        y_test: Test target data
        scaler: Scaler for denormalization

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict(X_test, verbose=0)

    # Denormalize
    y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1))
    pred_denorm = scaler.inverse_transform(predictions)

    # Calculate metrics
    mae = np.mean(np.abs(pred_denorm - y_test_denorm))
    rmse = np.sqrt(np.mean((pred_denorm - y_test_denorm) ** 2))
    mape = np.mean(np.abs((y_test_denorm - pred_denorm) / y_test_denorm)) * 100

    # Direction accuracy (did we predict the right direction?)
    if len(y_test) > 1:
        actual_direction = np.sign(y_test_denorm[1:] - y_test_denorm[:-1])
        pred_direction = np.sign(pred_denorm[1:] - pred_denorm[:-1])
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }
