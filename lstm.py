import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to load stock data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    return scaled_data, scaler

# Function to create dataset
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Streamlit App
st.title('Stock Price Prediction using LSTM')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):', 'AAPL')
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.button('Predict Stock Price'):
    data = load_data(ticker, start_date, end_date)
    st.write(f"Displaying stock data for {ticker}")
    st.write(data.tail())

    # Plot stock closing price
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Closing Price')
    ax.legend()
    st.pyplot(fig)
    
    # Preprocessing
    scaled_data, scaler = preprocess_data(data)
    time_step = 60
    X, Y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Train/Test Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    # Build LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, Y_train, epochs=10, batch_size=16, verbose=1)
    
    # Predict on test data
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # Visualizing predictions
    test_data = data.iloc[train_size + time_step + 1:]
    test_data['Predictions'] = predictions
    
    fig2, ax2 = plt.subplots()
    ax2.plot(test_data['Close'], label='Actual Price')
    ax2.plot(test_data['Predictions'], label='Predicted Price')
    ax2.legend()
    st.pyplot(fig2)
    
    st.write("Prediction Completed!")
