"""
Data Loader Module - Fetch historical stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS' for NSE stocks)
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)

        if hist.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in hist.columns:
                hist[col] = 0

        return hist[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def fetch_stock_data_by_date_range(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical stock data for a specific date range.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval

    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval=interval)

        if hist.empty:
            raise ValueError(f"No data found for ticker: {ticker} in date range")

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in hist.columns:
                hist[col] = 0

        return hist[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")


def get_stock_info(ticker: str) -> dict:
    """
    Get basic stock information.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with stock info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            'name': info.get('shortName', info.get('longName', ticker)),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'current_price': info.get('currentPrice', info.get('previousClose', 'N/A')),
            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
        }
    except Exception as e:
        return {'error': str(e)}


def cache_stock_data(ticker: str, data: pd.DataFrame, cache_dir: str = "data") -> None:
    """
    Cache stock data to disk.

    Args:
        ticker: Stock ticker symbol
        data: DataFrame with stock data
        cache_dir: Directory to save cached data
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save as CSV with ticker and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker.replace('.', '_')}_{timestamp}.csv"
    filepath = cache_path / filename

    data.to_csv(filepath)

    # Also save as latest
    latest_path = cache_path / f"{ticker.replace('.', '_')}_latest.csv"
    data.to_csv(latest_path)


def load_cached_data(ticker: str, cache_dir: str = "data") -> pd.DataFrame:
    """
    Load most recently cached data for a ticker.

    Args:
        ticker: Stock ticker symbol
        cache_dir: Directory containing cached data

    Returns:
        DataFrame with stock data or None if not found
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    # Find latest cached file for this ticker
    pattern = f"{ticker.replace('.', '_')}_latest.csv"
    latest_file = cache_path / pattern

    if latest_file.exists():
        return pd.read_csv(latest_file, index_col=0, parse_dates=True)

    return None


def refresh_data(ticker: str, cache_dir: str = "data") -> pd.DataFrame:
    """
    Fetch fresh data and cache it.

    Args:
        ticker: Stock ticker symbol
        cache_dir: Directory for caching

    Returns:
        DataFrame with fresh stock data
    """
    data = fetch_stock_data(ticker, period="2y")
    cache_stock_data(ticker, data, cache_dir)
    return data
