import yfinance as yf
import pandas as pd
import os


def load_stock_data(ticker="AAPL", period="5y", interval="1d"):
    print(f"📊 Downloading data for {ticker}...")

    data = yf.download(ticker, period=period, interval=interval)
    data.reset_index(inplace=True)

    data_dir = "data"

    if os.path.isfile(data_dir):
        raise Exception("❌ 'data' is a file. Delete it and create a folder.")

    os.makedirs(data_dir, exist_ok=True)

    file_path = f"{data_dir}/{ticker}_data.csv"
    data.to_csv(file_path, index=False)

    print(f"✅ Data saved at {file_path}")

    return data


if __name__ == "__main__":
    df = load_stock_data("AAPL")
    print(df.head())