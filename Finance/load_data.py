import yfinance as yf
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime
from tqdm import tqdm

# Configuration
start_date = "2021-01-01"
end_date = "2024-12-31"
output_dir = "financial_data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sp500_returns.pt")
vix_file = os.path.join(output_dir, "vix.pt")

# Fetch S&P 500 tickers
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_tickers = pd.read_html(sp500_url)[0]["Symbol"].tolist()
sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
# Limit to 100 tickers for testing; remove for full dataset
sp500_tickers = sp500_tickers[:100]

def download_stock(ticker):
    """Download stock data for a single ticker."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty:
            return ticker, data["Close"].values
        else:
            print(f"No data for {ticker}")
            return ticker, None
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
        return ticker, None

# Parallel download
print("Downloading stock data...")
stock_data = {}
dates = None
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(download_stock, ticker): ticker for ticker in sp500_tickers}
    for future in tqdm(as_completed(futures), total=len(sp500_tickers), desc="Stocks"):
        ticker, closes = future.result()
        if closes is not None:
            stock_data[ticker] = closes
            if dates is None:
                dates = pd.date_range(start=start_date, end=end_date, freq="B")[:len(closes)]

# Download VIX as condition
print("Downloading VIX data...")
vix_data = yf.download("^VIX", start=start_date, end=end_date, progress=False)
vix = vix_data["Close"].values if not vix_data.empty else np.zeros(len(dates))

# Convert to NumPy array
n_stocks = len(stock_data)
n_days = len(dates)
returns_array = np.zeros((n_stocks, n_days - 1), dtype=np.float32)  # Store returns
tickers = []

print("Calculating returns...")
for i, (ticker, closes) in enumerate(stock_data.items()):
    tickers.append(ticker)
    returns = (closes[1:] - closes[:-1]) / closes[:-1]  # Calculate returns
    returns_array[i] = (returns - returns.mean()) / returns.std()  # Standardize

# Convert to PyTorch tensor
returns_tensor = torch.tensor(returns_array, dtype=torch.float32)
vix_tensor = torch.tensor((vix[1:] - vix[1:].mean()) / vix[1:].std(), dtype=torch.float32)

# Save to .pt files
torch.save({"returns": returns_tensor, "tickers": tickers, "dates": dates[1:]}, output_file)
torch.save(vix_tensor, vix_file)

print(f"Data saved to {output_file} and {vix_file}")