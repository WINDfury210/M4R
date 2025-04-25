import yfinance as yf
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime
from tqdm import tqdm
import time
import pickle

# Configuration
start_date = "2019-01-01"  # 6 years
end_date = "2024-12-31"
output_dir = "financial_data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sp500_returns.pt")
vix_file = os.path.join(output_dir, "vix.pt")
cache_file = os.path.join(output_dir, "stock_data_cache.pkl")
dates_cache = os.path.join(output_dir, "dates_cache.pkl")

# Fetch S&P 500 tickers
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_tickers = pd.read_html(sp500_url)[0]["Symbol"].tolist()
sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]

def download_stock(ticker, retries=3):
    """Download stock data with retries."""
    for attempt in range(retries):
        try:
            ticker_obj = yf.Ticker(ticker)
            if not ticker_obj.info:
                print(f"Invalid ticker {ticker}")
                return ticker, None
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not data.empty and "Close" in data.columns:
                closes = data["Close"].iloc[:, 0].values if data["Close"].ndim > 1 else data["Close"].values
                closes = np.ravel(closes)
                if len(closes) > 0 and not np.any(np.isnan(closes)):
                    return ticker, closes
                print(f"Skipping {ticker}: contains NaN")
            return ticker, None
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait before retry
            continue
    print(f"Failed to download {ticker} after {retries} attempts")
    return ticker, None

# Load or download VIX
print("Downloading VIX data...")
if os.path.exists(dates_cache):
    dates = pickle.load(open(dates_cache, "rb"))
    vix_tensor = torch.load(vix_file, weights_only=False)
    print("Loaded cached VIX data")
else:
    vix_data = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False)
    if vix_data.empty:
        raise ValueError("Failed to download VIX data")
    vix = vix_data["Close"].values
    dates = vix_data.index
    vix_tensor = torch.tensor((vix[1:] - vix[1:].mean()) / vix[1:].std(), dtype=torch.float32)
    with open(dates_cache, "wb") as f:
        pickle.dump(dates, f)

n_days = len(dates)

# Load or download stocks
stock_data = {}
failed_tickers = []
if os.path.exists(cache_file):
    print("Loading cached stock data...")
    stock_data = pickle.load(open(cache_file, "rb"))
    failed_tickers = [t for t in sp500_tickers if t not in stock_data]
else:
    print("Downloading stock data...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_stock, ticker): ticker for ticker in sp500_tickers}
        for future in tqdm(as_completed(futures), total=len(sp500_tickers), desc="Stocks"):
            ticker, closes = future.result()
            if closes is not None:
                stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
                stock_df = stock_df.reindex(dates, method="ffill")
                closes = stock_df["Close"].iloc[:, 0].values if stock_df["Close"].ndim > 1 else stock_df["Close"].values
                closes = np.ravel(closes)
                if len(closes) == n_days and not np.any(np.isnan(closes)):
                    stock_data[ticker] = closes
                else:
                    print(f"Skipping {ticker}: length {len(closes)} != {n_days} or contains NaN")
                    failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
            time.sleep(2)  # Rate limit delay

    # Save cache
    with open(cache_file, "wb") as f:
        pickle.dump(stock_data, f)
    print(f"Saved stock data cache to {cache_file}")

if failed_tickers:
    print(f"Failed tickers: {failed_tickers}")
if not stock_data:
    raise ValueError("No valid stock data downloaded")

# Convert to NumPy array
n_stocks = len(stock_data)
returns_array = np.zeros((n_stocks, n_days - 1), dtype=np.float32)
tickers = []

print("Calculating returns...")
for i, (ticker, closes) in enumerate(stock_data.items()):
    tickers.append(ticker)
    if closes.shape[0] != n_days:
        print(f"Warning: {ticker} closes length {closes.shape[0]} != {n_days}, skipping")
        continue
    returns = (closes[1:] - closes[:-1]) / closes[:-1]
    returns = np.ravel(returns)
    if returns.shape[0] != n_days - 1:
        print(f"Warning: {ticker} returns length {returns.shape[0]} != {n_days-1}, skipping")
        continue
    returns_array[i] = (returns - np.mean(returns)) / np.std(returns)

# Convert to PyTorch tensor
returns_tensor = torch.tensor(returns_array, dtype=torch.float32)

# Save to .pt files
torch.save({"returns": returns_tensor, "tickers": tickers, "dates": dates[1:]}, output_file)
torch.save(vix_tensor, vix_file)

print(f"Data saved to {output_file} and {vix_file}")
print(f"Saved {returns_tensor.shape[0]} stocks with {returns_tensor.shape[1]} days")