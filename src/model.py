# LSTM model for stock price prediction


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def build_lstm_model(input_shape):
    """
    LSTM model for stock price prediction
    """

    print("🧠 Building LSTM model...")

    model = Sequential()

    # Layer 1
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Layer 2
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))

    # Dense layers
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))  # output = next price

    model.compile(optimizer="adam", loss="mean_squared_error")

    print("✅ Model built successfully!")

    return model