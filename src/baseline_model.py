"""
Baseline Models Module - Simple statistical models for comparison with deep learning
"""

import numpy as np
import pandas as pd
from typing import Tuple


class MovingAveragePredictor:
    """
    Simple Moving Average baseline predictor.
    Predicts future prices based on the average of recent prices.
    """

    def __init__(self, window: int = 20):
        """
        Initialize the moving average predictor.

        Args:
            window: Number of days to use for moving average
        """
        self.window = window
        self.fitted = False

    def fit(self, data: pd.DataFrame, feature: str = 'Close'):
        """Fit the predictor (calculate moving average)."""
        self.last_ma = data[feature].iloc[-self.window:].mean()
        self.trend = data[feature].iloc[-1] - data[feature].iloc[-self.window]
        self.fitted = True
        return self

    def predict(self, forecast_horizon: int) -> np.ndarray:
        """
        Generate predictions.

        Args:
            forecast_horizon: Number of days to predict

        Returns:
            Array of predicted prices
        """
        if not self.fitted:
            raise ValueError("Must call fit() before predict()")

        # Simple baseline: last MA + slight trend adjustment
        predictions = np.array([
            self.last_ma + (i * self.trend / self.window)
            for i in range(forecast_horizon)
        ])
        return predictions

    def get_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': 0  # Baseline doesn't track direction well
        }


class NaivePredictor:
    """
    Naive baseline predictor - predicts tomorrow's price = today's price.
    This is a common financial baseline (random walk hypothesis).
    """

    def __init__(self):
        self.last_price = None
        self.fitted = False

    def fit(self, data: pd.DataFrame, feature: str = 'Close'):
        """Fit the predictor."""
        self.last_price = data[feature].iloc[-1]
        self.fitted = True
        return self

    def predict(self, forecast_horizon: int) -> np.ndarray:
        """Predict all future days as the last known price."""
        if not self.fitted:
            raise ValueError("Must call fit() before predict()")

        return np.array([self.last_price] * forecast_horizon)

    def get_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': 0
        }


def compare_models(
    actual: np.ndarray,
    predictions: dict
) -> dict:
    """
    Compare multiple model predictions against actual values.

    Args:
        actual: Actual price values
        predictions: Dictionary with model names as keys and predictions as values

    Returns:
        Dictionary with comparison metrics for each model
    """
    comparison = {}

    for model_name, pred in predictions.items():
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100

        comparison[model_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }

    # Find best model
    best_model = min(comparison.keys(), key=lambda x: comparison[x]['mae'])
    comparison['best_model'] = best_model

    return comparison
