import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import os

# Instead of importing, define the functions directly in this cell

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
# ... (rest of the prediction code)


def predict_stock_prices(ticker='AAPL', start_date='2021-01-01', end_date='2021-12-31', look_back=60):
    # Load trained model and scaler
    try:
        model = tf.keras.models.load_model('models/best_model.h5')
    except:
        raise FileNotFoundError("No trained model found. Please run train.py first.")

    # Load new data for prediction
    data = load_stock_data(ticker, start_date, end_date)

    # We need the scaler used during training, but for simplicity we'll create a new one
    # In production, you should save and load the scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp_data = data['Close'].values.reshape(-1, 1)
    scaler.fit(temp_data)

    # Prepare test data
    x_test, dataset = prepare_test_data(data, scaler, look_back)

    # Make predictions
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot results
    plt.figure(figsize=(16, 8))
    plt.plot(data['Close'].values[look_back:], color='black', label=f'Actual {ticker} Price')
    plt.plot(predicted_prices, color='green', label=f'Predicted {ticker} Price')
    plt.title(f'{ticker} Share Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Share Price')
    plt.legend()
    plt.savefig('stock_prediction.png')
    plt.show()

    return predicted_prices

if __name__ == "__main__":
    predicted_prices = predict_stock_prices()
    print("Predictions completed. Check stock_prediction.png for results.")