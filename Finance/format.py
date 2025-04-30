from alpha_vantage.timeseries import TimeSeries
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
api_key = "YOUR_API_KEY"  # Replace with your Alpha Vantage API key

# Load S&P 500 tickers
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()[:400]

# Initialize Alpha Vantage
ts = TimeSeries(key=api_key, output_format='pandas')

# Download data
sequences = []
conditions = []
failed_tickers = []
for ticker in tickers:
    try:
        # Download stock data
        df, _ = ts.get_daily(symbol=ticker, outputsize='full')
        df = df.loc[start_date:end_date].sort_index()
        if df.empty or len(df) < sequence_length:
            failed_tickers.append(ticker)
            print(f"Skipping {ticker}: Empty or insufficient data")
            continue
        df.index = pd.to_datetime(df.index)
        returns = np.log(df['4. close'] / df['4. close'].shift(1)).dropna()
        
        # Download VIX (CBOE Volatility Index)
        vix_df, _ = ts.get_daily(symbol='VIX', outputsize='full')
        vix_df = vix_df.loc[start_date:end_date].sort_index()
        vix = vix_df['4. close'].reindex(returns.index, method='ffill')
        
        # Group by month
        monthly_starts = df.groupby([df.index.year, df.index.month]).head(1).index
        for start_date in monthly_starts:
            start_idx = returns.index.get_loc(start_date)
            if start_idx + sequence_length <= len(returns):
                seq = returns.iloc[start_idx:start_idx + sequence_length].values
                cond = vix.iloc[start_idx:start_idx + sequence_length].values
                if len(seq) == sequence_length and len(cond) == sequence_length:
                    sequences.append(seq)
                    conditions.append(cond)
        print(f"Processed {ticker}: {len(monthly_starts)} sequences")
    except Exception as e:
        failed_tickers.append(ticker)
        print(f"Failed to download {ticker}: {str(e)}")
    time.sleep(12)  # Alpha Vantage free tier: 5 requests/min

# Print failed tickers
if failed_tickers:
    print(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

# Convert to numpy array
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

# Verify statistics
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