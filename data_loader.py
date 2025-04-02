import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def load_stock_data(ticker, start_date, end_date):
    """
    Download stock data from Yahoo Finance
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data, look_back=60):
    """
    Preprocess data for LSTM model
    """
    # Use only 'Close' price
    dataset = data['Close'].values
    dataset = dataset.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create training dataset
    x_train, y_train = [], []

    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler, dataset

def prepare_test_data(data, scaler, look_back=60):
    """
    Prepare test data for prediction
    """
    dataset = data['Close'].values
    dataset = dataset.reshape(-1, 1)
    scaled_data = scaler.transform(dataset)

    x_test = []
    for i in range(look_back, len(scaled_data)):
        x_test.append(scaled_data[i-look_back:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test, dataset