import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# --- Model Loading (Adjust if you have a saved model) ---
def load_model():
    """
    Load and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(4, 1)))  
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = load_model()  

# --- Helper Functions ---
def get_display_date_range(option):
    """
    Get the display date range based on the user-selected option.
    """
    today = date.today()
    if option == "1 month":
        start = today - timedelta(days=30)
    elif option == "3 months":
        start = today - timedelta(days=90)
    elif option == "6 months":
        start = today - timedelta(days=180)
    elif option == "1 year":
        start = today - timedelta(days=365)
    elif option == "5 years":
        start = today - timedelta(days=1825)
    else:
        start = "2000-01-01"  # Using a fixed start date for 'max'
    return start, today

def download_data(ticker, start_date, end_date):
    """
    Download stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error("No data found for the given ticker symbol. Please enter a valid ticker symbol.")
        return None
    else:
        data["Date"] = data.index
        data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        data.reset_index(drop=True, inplace=True)
        return data

def preprocess_data(data):
    """
    Preprocess the data for model training.
    """
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[["Open", "High", "Low", "Volume"]])
    x = scaled_data
    y = data["Close"]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    xtrain = np.array(xtrain).reshape(xtrain.shape[0], xtrain.shape[1], 1)  
    xtest = np.array(xtest).reshape(xtest.shape[0], xtest.shape[1], 1)  
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)
    return xtrain, xtest, ytrain, ytest, scaler

def predict_prices(model, data, num_predictions, scaler):
    """
    Predict future stock prices using the trained model.
    """
    predictions = []
    features = np.array([data.iloc[-1, 1:5]], dtype=np.float32)
    features = scaler.transform(features)  # Normalize features
    features = features.reshape(1, features.shape[1], 1)
    for _ in range(num_predictions):
        prediction = model.predict(features)
        predictions.append(prediction[0][0])
        # Use the predicted value as the 'Close' for the next prediction
        new_feature = [features[0][0][0], features[0][1][0], features[0][2][0], prediction[0][0]]
        new_feature = np.array([new_feature], dtype=np.float32)
        new_feature = scaler.transform(new_feature)  # Normalize features
        features = new_feature.reshape(1, new_feature.shape[1], 1)
    return predictions

def plot_candlestick_chart(data, title):
    """
    Plot the candlestick chart for the stock data.
    """
    figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                            open=data["Open"],
                                            high=data["High"],
                                            low=data["Low"],
                                            close=data["Close"])])
    figure.update_layout(title=title, xaxis_rangeslider_visible=False)
    return figure

def plot_trend_chart(last_close_date, last_close_price, predictions, num_predictions):
    """
    Plot the predicted trend chart.
    """
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=[last_close_date], y=[last_close_price], mode='markers', name='Last Close'))
    trend_fig.add_trace(go.Scatter(x=[last_close_date + timedelta(days=i+1) for i in range(num_predictions)], 
                                   y=predictions, mode='lines+markers', name='Predicted Trend'))
    trend_fig.update_layout(title=f"Predicted Trend", xaxis_title='Date', yaxis_title='Price')
    return trend_fig

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def plot_correlation_matrix(data):
    """
    Plot the correlation matrix of the stock data.
    """
    corr_matrix = data[["Open", "High", "Low", "Close", "Volume"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title('Correlation Matrix')
    st.pyplot(fig)

def plot_moving_average(data, window):
    """
    Plot the moving average of the stock data.
    """
    data["MA"] = data["Close"].rolling(window=window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["MA"], mode='lines', name='Moving Average'))
    fig.update_layout(title="Moving Average", xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_bollinger_bands(data, window):
    """
    Plot the Bollinger Bands of the stock data.
    """
    data["MA"] = data["Close"].rolling(window=window).mean()
    data["BB_upper"] = data["MA"] + 2*data["Close"].rolling(window=window).std()
    data["BB_lower"] = data["MA"] - 2*data["Close"].rolling(window=window).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["BB_upper"], mode='lines', name='Upper Bollinger Band'))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["BB_lower"], mode='lines', name='Lower Bollinger Band'))
    fig.update_layout(title="Bollinger Bands", xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_volume_analysis(data):
    """
    Plot the volume of trades over time.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
    fig.update_layout(title="Volume Analysis", xaxis_title='Date', yaxis_title='Volume')
    return fig

# --- App Structure ---
st.title("Stock Price Prediction Web App")

# User inputs
ticker = st.text_input("Enter Stock Ticker", "")  # Allow user to input any stock ticker
start_date = st.date_input("Start Date", datetime.date(2021, 1, 1)) 
end_date = st.date_input("End Date", datetime.date.today())
display_range_option = st.selectbox("Select Display Date Range", ["1 month", "3 months", "6 months", "1 year", "5 years", "max"])
num_predictions = st.number_input("Number of Predictions", min_value=1, step=1, value=1)

# Advanced Data Visualization Options
st.markdown("### Advanced Data Visualization")
visualization_option = st.selectbox("Select Visualization Type", ["Moving Average", "Bollinger Bands", "Volume Analysis", "Correlation Matrix"])

if st.button("Predict"):
    if ticker.strip() == "":
        st.warning("Please enter a valid stock ticker.")
    else:
        try:
            # --- Data Collection and Preparation ---
            training_data = download_data(ticker, start_date, end_date)
            if training_data is not None:
                # --- Data Preprocessing ---
                xtrain, xtest, ytrain, ytest, scaler = preprocess_data(training_data)

                # Print the size of the training and testing datasets
                st.write(f"Training data size: {xtrain.shape[0]} samples")
                st.write(f"Testing data size: {xtest.shape[0]} samples")

                # --- Model Training (if needed) ---
                model.fit(xtrain, ytrain, epochs=10, batch_size=32, verbose=1)

                # --- Prediction ---
                predictions = predict_prices(model, training_data, num_predictions, scaler)
                st.write(f"Predicted Prices: {predictions}")

                # --- Data Visualization ---
                display_start_date, display_end_date = get_display_date_range(display_range_option)
                display_data = training_data[(training_data["Date"] >= pd.to_datetime(display_start_date)) & 
                                             (training_data["Date"] <= pd.to_datetime(display_end_date))]
                candlestick_chart = plot_candlestick_chart(display_data, f"{ticker} Stock Price Analysis")
                st.plotly_chart(candlestick_chart)

                last_close_date = training_data["Date"].iloc[-1]
                last_close_price = training_data["Close"].iloc[-1]
                trend_chart = plot_trend_chart(last_close_date, last_close_price, predictions, num_predictions)
                st.plotly_chart(trend_chart)

                # --- Additional Visualizations ---
                if visualization_option == "Moving Average":
                    window = st.slider("Select Moving Average Window", min_value=10, max_value=100, value=20)
                    if training_data is not None:
                        ma_chart = plot_moving_average(training_data, window)
                        st.plotly_chart(ma_chart)

                elif visualization_option == "Bollinger Bands":
                    window = st.slider("Select Bollinger Bands Window", min_value=10, max_value=100, value=20)
                    if training_data is not None:
                        bb_chart = plot_bollinger_bands(training_data, window)
                        st.plotly_chart(bb_chart)

                elif visualization_option == "Volume Analysis":
                    if training_data is not None:
                        volume_chart = plot_volume_analysis(training_data)
                        st.plotly_chart(volume_chart)

                elif visualization_option == "Correlation Matrix":
                    if training_data is not None:
                        plot_correlation_matrix(training_data)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
