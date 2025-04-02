import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# The following import was causing the error and is not needed
# from utils.data_loader import load_stock_data, preprocess_data
import matplotlib.pyplot as plt
import os

def build_model(look_back):
    """
    Build LSTM model
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_stock_model(ticker='AAPL', start_date='2010-01-01', end_date='2020-12-31', look_back=60):
    # Load and preprocess data
    data = load_stock_data(ticker, start_date, end_date)
    x_train, y_train, scaler, _ = preprocess_data(data, look_back)

    # Build model
    model = build_model(look_back)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, verbose=1),
        ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Model Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('training_loss.png')
    plt.close()

    return model, scaler

if __name__ == "__main__":
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)

    # Train model
    model, scaler = train_stock_model()
    print("Model training completed and saved to models/best_model.h5")