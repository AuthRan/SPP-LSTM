"""
Stock Market Prediction App - Streamlit Frontend for LSTM-based Stock Prediction
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import fetch_stock_data, get_stock_info, refresh_data
from src.preprocessing import prepare_data_for_training
from src.model import build_lstm_model, train_model, evaluate_model
from src.prediction import predict_with_existing_model
from src.sp500_tickers import get_nifty_50_tickers, get_ticker_name


# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None


def create_price_chart(data, predictions=None, confidence_bounds=None):
    """Create interactive price chart with predictions."""
    fig = go.Figure()

    # Historical prices
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
    ))

    # Predictions
    if predictions is not None:
        pred_dates = predictions['prediction_dates']
        pred_prices = predictions['predictions']

        # Prediction line
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=8),
            hovertemplate='<b>Date</b>: %{x}<br><b>Predicted</b>: ₹%{y:.2f}<extra></extra>'
        ))

        # Confidence interval
        if confidence_bounds is not None:
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions['upper_bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hovertemplate='Upper Bound: ₹%{y:.2f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions['lower_bound'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(width=0),
                name='95% Confidence Interval',
                hovertemplate='Lower Bound: ₹%{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title='Stock Price History & Prediction',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(tickprefix="₹", tickformat=".2f")

    return fig


def create_metrics_row(metrics):
    """Create a row of metrics."""
    cols = st.columns(4)

    with cols[0]:
        st.metric(
            label="Mean Absolute Error",
            value=f"₹{metrics.get('mae', 'N/A'):.2f}" if isinstance(metrics.get('mae'), (int, float)) else "N/A",
            help="Average absolute difference between predicted and actual prices"
        )

    with cols[1]:
        st.metric(
            label="RMSE",
            value=f"₹{metrics.get('rmse', 'N/A'):.2f}" if isinstance(metrics.get('rmse'), (int, float)) else "N/A",
            help="Root Mean Square Error - penalizes larger errors more"
        )

    with cols[2]:
        st.metric(
            label="MAPE",
            value=f"{metrics.get('mape', 'N/A'):.2f}%" if isinstance(metrics.get('mape'), (int, float)) else "N/A",
            help="Mean Absolute Percentage Error"
        )

    with cols[3]:
        direction = metrics.get('direction_accuracy', 0)
        st.metric(
            label="Direction Accuracy",
            value=f"{direction:.1f}%" if isinstance(direction, (int, float)) else "N/A",
            help="Accuracy of predicting price direction (up/down)"
        )


def main():
    """Main application."""
    initialize_session_state()

    # Header
    st.markdown('<p class="main-header">📈 Stock Price Predictor</p>', unsafe_allow_html=True)
    st.markdown("### LSTM-based prediction for Nifty 50 stocks")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Controls")

    # Ticker selection
    tickers = get_nifty_50_tickers()
    selected_ticker = st.sidebar.selectbox(
        "Select Stock",
        options=tickers,
        format_func=lambda x: f"{get_ticker_name(x)} ({x.replace('.NS', '')})",
        help="Choose a Nifty 50 stock to analyze"
    )

    # Settings
    st.sidebar.subheader("Model Settings")
    sequence_length = st.sidebar.slider(
        "Lookback Window (days)",
        min_value=30,
        max_value=120,
        value=60,
        step=10,
        help="Number of past days to use for prediction"
    )
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Number of future days to predict"
    )
    epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        help="Number of training iterations"
    )

    # Action buttons
    st.sidebar.subheader("Actions")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        load_data_btn = st.button("📊 Load Data", use_container_width=True)

    with col2:
        train_model_btn = st.button("🤖 Train Model", use_container_width=True)

    predict_btn = st.sidebar.button("🔮 Get Prediction", use_container_width=True)

    # Fetch stock info
    if selected_ticker:
        with st.spinner(f"Fetching info for {get_ticker_name(selected_ticker)}..."):
            stock_info = get_stock_info(selected_ticker)

    # Helper function to load data with sufficient history
    def load_historical_data(ticker, period="5y"):
        try:
            return fetch_stock_data(ticker, period=period)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

        # Display stock info
        st.sidebar.markdown("---")
        st.sidebar.subheader("Stock Info")

        if stock_info and 'error' not in stock_info:
            st.sidebar.write(f"**Company**: {stock_info.get('name', 'N/A')}")
            st.sidebar.write(f"**Sector**: {stock_info.get('sector', 'N/A')}")
            current_price = stock_info.get('current_price', 'N/A')
            if current_price != 'N/A':
                st.sidebar.metric("Current Price", f"₹{current_price:.2f}")
        else:
            st.sidebar.warning("Could not fetch stock info")

    # Main content area
    if load_data_btn:
        with st.spinner("Loading historical data (5 years for better training)..."):
            try:
                stock_data = fetch_stock_data(selected_ticker, period="5y")
                if len(stock_data) < 80:
                    st.warning(f"Only {len(stock_data)} days of data available. Consider using a different stock.")
                st.session_state.stock_data = stock_data
                st.session_state.current_ticker = selected_ticker
                st.success(f"Loaded {len(stock_data)} days of data for {get_ticker_name(selected_ticker)}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Display stock data chart
    if st.session_state.stock_data is not None:
        st.subheader(f"{get_ticker_name(selected_ticker)} ({selected_ticker})")

        # Price chart
        st.plotly_chart(
            create_price_chart(st.session_state.stock_data),
            use_container_width=True
        )

        # Show data statistics
        col1, col2, col3, col4 = st.columns(4)
        data = st.session_state.stock_data

        with col1:
            st.metric("Latest Close", f"₹{data['Close'].iloc[-1]:.2f}")

        with col2:
            period_high = data['High'].max()
            st.metric("Period High", f"₹{period_high:.2f}")

        with col3:
            period_low = data['Low'].min()
            st.metric("Period Low", f"₹{period_low:.2f}")

        with col4:
            avg_volume = data['Volume'].mean()
            st.metric("Avg Volume", f"{avg_volume:,.0f}")

    # Train model
    if train_model_btn and st.session_state.stock_data is not None:
        with st.spinner("Training LSTM model..."):
            try:
                # Prepare data
                prepared = prepare_data_for_training(
                    st.session_state.stock_data,
                    sequence_length=sequence_length,
                    forecast_horizon=forecast_horizon,
                    train_ratio=0.8
                )

                # Build model
                model = build_lstm_model(
                    sequence_length=sequence_length,
                    units=50,
                    dropout_rate=0.2,
                    forecast_horizon=1
                )

                # Train
                trained_model, history, train_metrics = train_model(
                    model,
                    prepared['X_train'],
                    prepared['y_train'],
                    prepared['X_test'],
                    prepared['y_test'],
                    epochs=epochs,
                    batch_size=32,
                    ticker=selected_ticker
                )

                # Evaluate
                eval_metrics = evaluate_model(
                    trained_model,
                    prepared['X_test'],
                    prepared['y_test'],
                    prepared['scaler']
                )

                st.session_state.model_trained = True
                st.session_state.current_model = trained_model
                st.session_state.scaler = prepared['scaler']

                st.success("Model trained successfully!")

                # Show training metrics
                st.subheader("Training Metrics")
                create_metrics_row(eval_metrics)

                # Training history chart
                st.subheader("Training History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history.history['loss'],
                    name='Training Loss',
                    line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation Loss',
                    line=dict(color='#ff7f0e')
                ))
                fig.update_layout(
                    title='Model Loss During Training',
                    xaxis_title='Epoch',
                    yaxis_title='Loss (MSE)',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.info("Tip: Make sure you have loaded data first")

    # Get predictions
    if predict_btn:
        if st.session_state.stock_data is not None:
            with st.spinner("Generating predictions..."):
                try:
                    # Try to use existing trained model
                    if st.session_state.get('model_trained'):
                        model = st.session_state.current_model
                        scaler = st.session_state.scaler

                        from src.prediction import predict_future_prices
                        predictions, confidence_bounds = predict_future_prices(
                            model,
                            st.session_state.stock_data,
                            scaler,
                            sequence_length=sequence_length,
                            forecast_horizon=forecast_horizon
                        )

                        st.session_state.predictions = {
                            'predictions': predictions,
                            'lower_bound': confidence_bounds[0],
                            'upper_bound': confidence_bounds[1],
                            'prediction_dates': pd.date_range(
                                start=st.session_state.stock_data.index[-1] + pd.Timedelta(days=1),
                                periods=forecast_horizon
                            )
                        }
                    else:
                        # Try to load pre-trained model from disk
                        result = predict_with_existing_model(
                            selected_ticker,
                            st.session_state.stock_data,
                            sequence_length=sequence_length,
                            forecast_horizon=forecast_horizon
                        )

                        if 'error' not in result:
                            st.session_state.predictions = result
                        else:
                            st.warning("No trained model found. Please train a model first.")

                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
        else:
            st.warning("Please load stock data first")

    # Display predictions
    if st.session_state.predictions is not None:
        st.markdown("---")
        st.subheader("🔮 Price Predictions")

        pred = st.session_state.predictions

        # Chart with predictions
        fig = create_price_chart(
            st.session_state.stock_data,
            pred,
            (pred['lower_bound'], pred['upper_bound'])
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prediction table
        st.subheader("Prediction Details")

        pred_df = pd.DataFrame({
            'Date': pred['prediction_dates'],
            'Predicted Price': pred['predictions'],
            'Lower Bound': pred['lower_bound'],
            'Upper Bound': pred['upper_bound']
        })

        pred_df['Date'] = pred_df['Date'].dt.strftime('%Y-%m-%d')
        pred_df['Predicted Price'] = pred_df['Predicted Price'].round(2)
        pred_df['Lower Bound'] = pred_df['Lower Bound'].round(2)
        pred_df['Upper Bound'] = pred_df['Upper Bound'].round(2)

        st.dataframe(pred_df, use_container_width=True)

        # Trend indicator
        first_pred = pred['predictions'][0]
        last_pred = pred['predictions'][-1]
        last_actual = pred.get('last_historical_price', st.session_state.stock_data['Close'].iloc[-1])

        col1, col2, col3 = st.columns(3)

        with col1:
            change = ((first_pred - last_actual) / last_actual) * 100
            delta = f"{change:+.2f}%"
            st.metric(
                "Day 1 vs Current",
                f"₹{first_pred:.2f}",
                delta=delta,
                delta_color="normal" if change >= 0 else "inverse"
            )

        with col2:
            mid_change = ((pred['predictions'][len(pred['predictions'])//2] - last_actual) / last_actual) * 100
            st.metric(
                "Mid-Period Prediction",
                f"₹{pred['predictions'][len(pred['predictions'])//2]:.2f}",
                delta=f"{mid_change:+.2f}%",
                delta_color="normal" if mid_change >= 0 else "inverse"
            )

        with col3:
            total_change = ((last_pred - last_actual) / last_actual) * 100
            st.metric(
                "End of Period",
                f"₹{last_pred:.2f}",
                delta=f"{total_change:+.2f}%",
                delta_color="normal" if total_change >= 0 else "inverse"
            )

        # Trend summary
        if last_pred > first_pred:
            st.success(f"📈 **Bullish Trend**: Price expected to increase by {((last_pred - first_pred) / first_pred) * 100:.2f}% over {forecast_horizon} days")
        else:
            st.warning(f"📉 **Bearish Trend**: Price expected to decrease by {((first_pred - last_pred) / first_pred) * 100:.2f}% over {forecast_horizon} days")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Disclaimer:</strong> This tool uses LSTM deep learning for educational purposes only.
        Stock market predictions are inherently uncertain. Past performance does not guarantee future results.
        Always consult a financial advisor before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
