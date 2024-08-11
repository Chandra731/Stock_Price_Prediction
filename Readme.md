# Stock Price Prediction Web App

This web application predicts stock prices using a Long Short-Term Memory (LSTM) model. The app fetches historical stock data from Yahoo Finance and allows users to visualize data trends and predict future prices.

## Features

- Download and preprocess stock data
- Visualize stock data with candlestick charts, moving averages, Bollinger Bands, and more
- Train an LSTM model to predict future stock prices
- View predicted stock trends
- Analyze trading volume and correlation between different stock features

## Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **yFinance**
- **Scikit-learn**
- **Keras**
- **Plotly**

## Installation

1. Clone the repository:
    git clone https://github.com/chandra731/Stock_Price_Prediction.git
2. Navigate to the project directory:   
    cd stock-price-prediction-app
3. Install required packages:   
    pip install -r requirements.txt

## Usage

1. Run the Streamlit app:
    streamlit run app.py
2. Enter the stock ticker and other required details to view predictions and visualizations.

## License

This project is licensed under the MIT License. See the LICENCE.md file for details.
