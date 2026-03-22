"""
Preprocessing Module - Data normalization and sequence creation for LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


def normalize_data(data: pd.DataFrame, feature: str = 'Close') -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize stock price data using MinMaxScaler.

    Args:
        data: DataFrame with stock data
        feature: Column to normalize (default: 'Close')

    Returns:
        Tuple of (normalized_data, scaler)
    """
    # Extract the feature column
    prices = data[[feature]].values

    # Initialize and fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    return scaled_prices, scaler


def denormalize_data(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Convert normalized data back to original scale.

    Args:
        scaled_data: Normalized data array
        scaler: Fitted MinMaxScaler

    Returns:
        Denormalized data
    """
    return scaler.inverse_transform(scaled_data)


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 60,
    forecast_horizon: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.

    Args:
        data: Normalized price data (1D array)
        sequence_length: Number of past days to use as input (lookback window)
        forecast_horizon: Number of future days to predict

    Returns:
        Tuple of (X, y) where:
            X: Input sequences of shape (num_samples, sequence_length, 1)
            y: Target values (next day's price) of shape (num_samples, 1)
    """
    X = []
    y = []

    # Create sequences where each sample uses past 'sequence_length' days
    # to predict the next day's price
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i + sequence_length])
        # Predict the price 'forecast_horizon' days ahead
        y.append(data[i + sequence_length + forecast_horizon - 1])

    return np.array(X), np.array(y)


def create_sequences_multi_output(
    data: np.ndarray,
    sequence_length: int = 60,
    forecast_horizon: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for multi-day forecasting.

    Args:
        data: Normalized price data
        sequence_length: Lookback window
        forecast_horizon: Number of days to predict

    Returns:
        Tuple of (X, y) where y contains 'forecast_horizon' future values
    """
    X = []
    y = []

    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])

    return np.array(X), np.array(y)


def train_test_split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data for time series (no shuffling to preserve temporal order).

    Args:
        X: Input sequences
        y: Target values
        train_ratio: Fraction of data for training

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    split_idx = int(len(X) * train_ratio)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test


def prepare_data_for_training(
    df: pd.DataFrame,
    feature: str = 'Close',
    sequence_length: int = 60,
    forecast_horizon: int = 10,
    train_ratio: float = 0.8
) -> dict:
    """
    Complete data preparation pipeline for LSTM training.

    Args:
        df: DataFrame with OHLCV data
        feature: Column to use for prediction
        sequence_length: Lookback window size
        forecast_horizon: Days to predict
        train_ratio: Training data ratio

    Returns:
        Dictionary with prepared data and scaler:
            - X_train, y_train, X_test, y_test
            - scaler: For denormalization
            - original_data: Raw price data
    """
    # Normalize data
    scaled_data, scaler = normalize_data(df, feature=feature)

    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length, forecast_horizon)

    if len(X) == 0:
        min_required = sequence_length + forecast_horizon
        raise ValueError(
            f"Insufficient data for training. Need at least {min_required} days, "
            f"but only have {len(df)} days. Please fetch more historical data."
        )

    # Split data
    X_train, y_train, X_test, y_test = train_test_split_time_series(X, y, train_ratio)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'original_data': df[[feature]].values,
        'scaled_data': scaled_data
    }


def get_feature_columns() -> List[str]:
    """Return list of available features for prediction."""
    return ['Open', 'High', 'Low', 'Close', 'Volume']
