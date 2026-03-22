# Stock Market Prediction System 📈

LSTM-based stock price prediction system for Nifty 50 stocks with an interactive Streamlit dashboard.

## Features

- **Data Fetching**: Real-time historical data from Yahoo Finance for NSE stocks
- **LSTM Deep Learning**: Recurrent neural network for time series prediction
- **Interactive Dashboard**: Streamlit-based UI for exploration and prediction
- **Nifty 50 Coverage**: All 50 stocks from India's benchmark index
- **Visualizations**: Interactive Plotly charts with confidence intervals
- **Model Metrics**: MAE, RMSE, MAPE, and direction accuracy

## Screenshots
<img width="1511" height="830" alt="image" src="https://github.com/user-attachments/assets/731e163a-81f5-47f4-9b74-d52cd790f5a2" />
<img width="1479" height="657" alt="image" src="https://github.com/user-attachments/assets/da800212-79c2-4520-be92-84b276243dc3" />
<img width="1445" height="734" alt="image" src="https://github.com/user-attachments/assets/3fba59be-68d7-4cfa-b21e-f92ae8097013" />


## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (already set up)

### Setup

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start the Dashboard

```bash
streamlit run app.py
```

### Workflow

1. **Select a Stock**: Choose from Nifty 50 tickers in the sidebar
2. **Load Data**: Click "Load Data" to fetch 2 years of historical data
3. **Train Model**: Click "Train Model" to train the LSTM
4. **Get Prediction**: Click "Get Prediction" to see future price forecasts

### Configuration Options

- **Lookback Window**: Number of past days used for prediction (default: 60)
- **Forecast Horizon**: Number of future days to predict (default: 10)
- **Training Epochs**: Number of training iterations (default: 50)

## Project Structure

```
Try_Finance/
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Yahoo Finance data fetching
│   ├── preprocessing.py     # Data normalization & sequences
│   ├── model.py             # LSTM model architecture & training
│   ├── prediction.py        # Prediction generation
│   └── sp500_tickers.py     # Nifty 50 ticker list
├── data/                     # Cached stock data
└── models/                   # Saved LSTM models
```

## Nifty 50 Stocks Included

| Ticker | Company | Ticker | Company |
|--------|---------|--------|---------|
| RELIANCE.NS | Reliance Industries | TCS.NS | Tata Consultancy Services |
| HDFCBANK.NS | HDFC Bank | INFY.NS | Infosys |
| ICICIBANK.NS | ICICI Bank | HINDUNILVR.NS | Hindustan Unilever |
| SBIN.NS | State Bank of India | BHARTIARTL.NS | Bharti Airtel |
| ... and 40 more stocks | | | |

## Model Architecture

```
Input: (60, 1) - 60 days lookback
├── LSTM(50, return_sequences=True)
├── Dropout(0.2)
├── LSTM(50)
├── Dropout(0.2)
├── Dense(25, relu)
└── Dense(1) - Output: predicted price
```

## Metrics Explained

- **MAE**: Mean Absolute Error - average prediction error in rupees
- **RMSE**: Root Mean Square Error - penalizes larger errors
- **MAPE**: Mean Absolute Percentage Error - relative error
- **Direction Accuracy**: How often the model predicts correct direction (up/down)

## Disclaimer

⚠️ **For Educational Purposes Only**

Stock market predictions are inherently uncertain. This tool uses historical patterns and cannot predict:
- Black swan events
- Policy changes
- Market sentiment shifts
- Global economic shocks

Always consult a SEBI-registered financial advisor before making investment decisions.

## License

MIT License
