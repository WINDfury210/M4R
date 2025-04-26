import yfinance as yf
import pandas as pd
import torch
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
sequence_length = 252
start_date = "2019-01-01"
end_date = "2024-12-31"
output_dir = "financial_data/sequences"
os.makedirs(output_dir, exist_ok=True)

# Load S&P 500 tickers
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist() # Fix ticker format, increase to 200

# Download data
sequences = []
conditions = []
failed_tickers = []
for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty or len(df) < sequence_length:
            failed_tickers.append(ticker)
            continue
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        vix = vix.reindex(returns.index, method='ffill').values
        for i in range(0, len(returns) - sequence_length + 1, 10):
            seq = returns[i:i+sequence_length].values
            cond = vix[i:i+sequence_length]
            if len(seq) == sequence_length and len(cond) == sequence_length:
                sequences.append(seq)
                conditions.append(cond)
    except Exception as e:
        failed_tickers.append(ticker)
        print(f"Failed to download {ticker}: {str(e)}")
        continue

# Print failed tickers
if failed_tickers:
    print(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

# Convert to numpy array first
sequences = np.array(sequences)
conditions = np.array(conditions)

# Convert to tensor
sequences = torch.tensor(sequences, dtype=torch.float32)
conditions = torch.tensor(conditions, dtype=torch.float32)

# Standardize per sequence
sequences = (sequences - sequences.mean(dim=1, keepdim=True)) / (sequences.std(dim=1, keepdim=True) + 1e-8)
sequences = sequences * 0.015  # Scale to typical std
conditions = (conditions - conditions.mean()) / (conditions.std() + 1e-8)  # Normalize VIX

# Save
output_file = f"{output_dir}/sequences_{sequence_length}.pt"
torch.save({"sequences": sequences, "conditions": conditions}, output_file)
print(f"Saved {len(sequences)} sequences to {output_file}")

# Verify statistics
print(f"Sequences mean: {sequences.mean().item():.6f}")
print(f"Sequences std: {sequences.std().item():.6f}")
print(f"Sequences shape: {sequences.shape}")
print(f"Conditions mean: {conditions.mean().item():.6f}")
print(f"Conditions std: {conditions.std().item():.6f}")