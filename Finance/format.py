import yfinance as yf
import pandas as pd
import torch
import numpy as np
import os
from datetime import datetime
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
sequence_length = 252
start_date = "2019-01-01"
end_date = "2024-12-31"
output_dir = "financial_data/sequences"
os.makedirs(output_dir, exist_ok=True)

# Load S&P 500 tickers
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()[:400]

# Download data
sequences = []
conditions = []
failed_tickers = []
batch_size = 50

# Download VIX first
try:
    vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)
    if vix_df.empty:
        raise ValueError("VIX data is empty")
    print("VIX data downloaded successfully")
except Exception as e:
    print(f"Failed to download ^VIX: {str(e)}")
    exit(1)

for i in range(0, len(tickers), batch_size):
    batch_tickers = tickers[i:i + batch_size]
    for ticker in batch_tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df.empty or len(df) < sequence_length:
                failed_tickers.append(ticker)
                print(f"Skipping {ticker}: Empty or insufficient data")
                continue
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            vix = vix_df['Close'].reindex(returns.index, method='ffill')
            
            # Group by month to get first trading day
            df.index = pd.to_datetime(df.index)
            monthly_starts = df.groupby([df.index.year, df.index.month]).head(1).index
            for start_date in monthly_starts:
                start_idx = returns.index.get_loc(start_date)
                if start_idx + sequence_length <= len(returns):
                    seq = returns.iloc[start_idx:start_idx + sequence_length].values
                    cond = vix.iloc[start_idx:start_idx + sequence_length].values
                    if len(seq) == sequence_length and len(cond) == sequence_length:
                        sequences.append(seq)
                        conditions.append(cond)
            print(f"Processed {ticker} successfully")
        except Exception as e:
            failed_tickers.append(ticker)
            print(f"Failed to download {ticker}: {str(e)}")
        time.sleep(1)  # Avoid rate limiting

# Print failed tickers
if failed_tickers:
    print(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

# Convert to numpy array and remove extra dimension
sequences = np.array(sequences)
sequences = np.squeeze(sequences)
conditions = np.array(conditions)
conditions = np.squeeze(conditions)

# Debug shapes
print(f"Sequences numpy shape: {sequences.shape}")
print(f"Conditions numpy shape: {conditions.shape}")

# Convert to tensor
sequences = torch.tensor(sequences, dtype=torch.float32)
conditions = torch.tensor(conditions, dtype=torch.float32)

# Debug shapes
print(f"Sequences tensor shape: {sequences.shape}")
print(f"Conditions tensor shape: {conditions.shape}")

# Verify shape and statistics before saving
print(f"Sequences shape before saving: {sequences.shape}")
print(f"Sequences mean: {sequences.mean().item():.6f}")
print(f"Sequences std: {sequences.std().item():.6f}")
print(f"Conditions mean: {conditions.mean().item():.6f}")
print(f"Conditions std: {conditions.std().item():.6f}")
assert sequences.dim() == 2, f"Expected 2D tensor, got shape {sequences.shape}"
assert sequences.shape[1] == sequence_length, f"Expected sequence length {sequence_length}, got {sequences.shape[1]}"

# Save
output_file = f"{output_dir}/sequences_{sequence_length}.pt"
torch.save({"sequences": sequences, "conditions": conditions}, output_file)
print(f"Saved {len(sequences)} sequences to {output_file}")