from flask import Flask, render_template, request
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from src.data_loader import load_stock_data
from src.preprocess import preprocess_data, create_sequences

app = Flask(__name__)

model = load_model("stock_lstm_model.h5")


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    error = None
    graph = None

    if request.method == "POST":

        ticker = request.form.get("ticker")

        try:
            # 📊 Load data
            df = load_stock_data(ticker)

            # 🧹 Preprocess
            df = preprocess_data(df)

            data = df[["Close"]]

            # 🔄 Create sequences (30 window)
            X, y = create_sequences(data, "Close", window_size=30)

            # 🛑 safety check
            if len(X) == 0:
                error = "Not enough data (need 30+ records)"
                return render_template("index.html", prediction=None, error=error)

            # 🧠 Prediction
            pred = model.predict(X[-1].reshape(1, 30, 1))
            prediction = float(pred[0][0])

            # 📈 GRAPH
            plt.figure(figsize=(7,4))
            plt.plot(y[-50:], label="Actual Price", color="blue")
            plt.title(f"{ticker} Stock Trend")
            plt.legend()

            # convert graph to image
            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            graph = base64.b64encode(img.getvalue()).decode()
            plt.close()

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        graph=graph
    )


if __name__ == "__main__":
    app.run(debug=True)