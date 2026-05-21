import matplotlib
matplotlib.use('TkAgg')
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from data_loader import load_stock_data
from preprocess import preprocess_data, create_sequences
from model import build_lstm_model
from sklearn.model_selection import train_test_split


# 1. Load data
df = load_stock_data("AAPL")

# 2. Preprocess
df = preprocess_data(df)

# 3. Prepare data
data = df[["Close"]]

X, y = create_sequences(data, "Close", window_size=60)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 5. Build model
model = build_lstm_model((X.shape[1], 1))

# 6. Train model
print("🚀 Training started...")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 7. Save model
model.save("stock_lstm_model.h5")

print("✅ Model trained and saved!")



# HOW IT WORKS (SIMPLE)
# Past 60 days stock → LSTM → Next day price



# 1. training code
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 2. prediction on test data
predictions = model.predict(X_test)

# 3. SAVE MODEL
model.save("stock_lstm_model.h5")

print("✅ Model trained and saved!")

# ============================
# 🔥 ADD THIS BELOW (IMPORTANT)
# ============================

# 📊 Next day prediction
next_day = model.predict(X[-1].reshape(1, 60, 1))
print("📈 Next Day Price:", next_day)

# 📉 Error calculation
from sklearn.metrics import mean_squared_error
print("📊 MSE:", mean_squared_error(y_test, predictions))