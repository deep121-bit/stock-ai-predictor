import numpy as np
import pandas as pd


def preprocess_data(df):
    print("🧹 Cleaning data...")

    df = df.sort_values("Date")
    df = df.dropna()

    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Close"].rolling(window=7).std()

    df = df.dropna()

    print("✅ Feature engineering done!")
    return df


def create_sequences(data, target_col="Close", window_size=60):

    values = data[target_col].values

    # 🛑 SAFE CHECK
    if len(values) <= window_size:
        return np.array([]), np.array([])

    X = []
    y = []

    for i in range(window_size, len(values)):
        X.append(values[i-window_size:i])
        y.append(values[i])

    X = np.array(X)
    y = np.array(y)

    # safety reshape
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y





# LSTM ko:

'''time pattern chahiye ⏳
sequence structure chahiye 📊

Ye step:
✔ Time dependency build karta hai
✔ Deep learning input ready karta hai'''