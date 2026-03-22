"""
Prediction Module - Load trained models and generate stock price predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .model import load_trained_model, get_default_model_path
from .preprocessing import normalize_data, create_sequences


def predict_future_prices(
    model,
    data: pd.DataFrame,
    scaler,
    sequence_length: int = 60,
    forecast_horizon: int = 10,
    feature: str = 'Close'
) -> tuple:
    """
    Predict future stock prices using a trained LSTM model.

    Args:
        model: Trained LSTM model
        data: Historical stock data DataFrame
        scaler: Fitted MinMaxScaler
        sequence_length: Lookback window size
        forecast_horizon: Number of days to predict
        feature: Feature column used for prediction

    Returns:
        Tuple of (predictions, confidence_bounds)
            predictions: Array of predicted prices
            confidence_bounds: Tuple of (lower_bound, upper_bound) arrays
    """
    # Normalize the data
    scaled_data, _ = normalize_data(data, feature=feature)

    # Use the last 'sequence_length' days to predict future
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

    predictions = []

    # Generate predictions for forecast_horizon days
    current_sequence = last_sequence.copy()

    for _ in range(forecast_horizon):
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])

        # Update sequence (shift and add new prediction)
        new_seq = np.roll(current_sequence[0], -1, axis=0)
        new_seq[-1, 0] = pred[0, 0]
        current_sequence = new_seq.reshape(1, sequence_length, 1)

    # Convert to numpy array
    predictions = np.array(predictions).reshape(-1, 1)

    # Denormalize predictions
    denorm_predictions = scaler.inverse_transform(predictions)

    # Calculate confidence bounds (using standard deviation of recent prices)
    recent_std = np.std(scaled_data[-sequence_length:])
    confidence_interval = recent_std * scaler.scale_[0] * 1.96  # 95% CI

    lower_bound = denorm_predictions.flatten() - confidence_interval
    upper_bound = denorm_predictions.flatten() + confidence_interval

    return denorm_predictions.flatten(), (lower_bound, upper_bound)


def predict_with_existing_model(
    ticker: str,
    data: pd.DataFrame,
    model_path: str = None,
    model_dir: str = "models",
    sequence_length: int = 60,
    forecast_horizon: int = 10,
    feature: str = 'Close'
) -> dict:
    """
    Load a saved model and generate predictions.

    Args:
        ticker: Stock ticker symbol
        data: Historical stock data
        model_path: Path to saved model (optional, will auto-find if None)
        model_dir: Directory containing saved models
        sequence_length: Lookback window
        forecast_horizon: Days to predict
        feature: Feature to predict

    Returns:
        Dictionary with predictions and metadata
    """
    # Find model path
    if model_path is None:
        model_path = get_default_model_path(ticker, model_dir)

    if model_path is None or not Path(model_path).exists():
        return {'error': f"No trained model found for ticker: {ticker}"}

    # Load model
    model = load_trained_model(model_path)

    # Normalize data
    scaled_data, scaler = normalize_data(data, feature=feature)

    # Generate predictions
    predictions, confidence_bounds = predict_future_prices(
        model, data, scaler, sequence_length, forecast_horizon, feature
    )

    # Get prediction dates
    last_date = data.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    return {
        'ticker': ticker,
        'model_path': model_path,
        'predictions': predictions,
        'lower_bound': confidence_bounds[0],
        'upper_bound': confidence_bounds[1],
        'prediction_dates': prediction_dates,
        'last_historical_price': data[feature].iloc[-1],
        'forecast_horizon': forecast_horizon
    }


def calculate_prediction_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray = None
) -> dict:
    """
    Calculate metrics for predictions.

    Args:
        predictions: Predicted prices
        actuals: Actual prices (optional, for comparison)

    Returns:
        Dictionary with metrics
    """
    metrics = {
        'mean_prediction': np.mean(predictions),
        'std_prediction': np.std(predictions),
        'min_prediction': np.min(predictions),
        'max_prediction': np.max(predictions),
        'trend': 'bullish' if predictions[-1] > predictions[0] else 'bearish'
    }

    if actuals is not None:
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        metrics.update({
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })

    return metrics


def get_prediction_summary(
    ticker: str,
    predictions: np.ndarray,
    prediction_dates: pd.DatetimeIndex,
    last_price: float
) -> str:
    """
    Generate a human-readable prediction summary.

    Args:
        ticker: Stock ticker
        predictions: Predicted prices
        prediction_dates: Dates of predictions
        last_price: Last historical price

    Returns:
        Summary string
    """
    first_pred = predictions[0]
    last_pred = predictions[-1]

    change_from_last = ((first_pred - last_price) / last_price) * 100
    total_change = ((last_pred - first_pred) / first_pred) * 100

    trend = "upward" if last_pred > first_pred else "downward"

    summary = f"""
    Stock: {ticker}
    Current Price: ₹{last_price:.2f}

    {forecast_horizon}-Day Prediction:
    - Starting at: ₹{first_pred:.2f} ({change_from_last:+.2f}%)
    - Ending at: ₹{last_pred:.2f}
    - Overall trend: {trend} ({total_change:+.2f}%)
    """

    return summary
