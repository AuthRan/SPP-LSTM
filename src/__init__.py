"""
Stock Prediction Package - LSTM-based stock price prediction system
"""

from .data_loader import fetch_stock_data, get_stock_info, refresh_data
from .preprocessing import prepare_data_for_training, normalize_data, create_sequences
from .model import build_lstm_model, train_model, load_trained_model
from .prediction import predict_future_prices, predict_with_existing_model
from .sp500_tickers import get_nifty_50_tickers, get_ticker_name

__version__ = "1.0.0"
__author__ = "Stock Prediction System"
