import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_stock_data
from preprocess import preprocess_data, create_sequences


# 1. Load data
df = load_stock_data("AAPL")
df = preprocess_data(df)

data = df[["Close"]]

# 2. Create sequences
X, y = create_sequences(data, "Close", window_size=60)

# 3. Load trained model
model = load_model("stock_lstm_model.h5")

print("📊 Making predictions...")

# 4. Predict
predictions = model.predict(X)

# 5. Reshape for plotting
predictions = predictions.reshape(-1)
y = y.reshape(-1)

# 6. Plot results
plt.figure(figsize=(12,6))
plt.plot(y, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("📈 Stock Price Prediction (AAPL)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

plt.show()