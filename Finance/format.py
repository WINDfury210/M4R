import yfinance as yf
import pandas as pd
import torch
import numpy as np
import os

# Configuration
sequence_length = 252
start_date = "2019-01-01"
end_date = "2024-12-31"
output_dir = "financial_data/sequences"
os.makedirs(output_dir, exist_ok=True)

# Load S&P 500 tickers
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].tolist()[:100]  # First 100 for simplicity

# Download data
sequences = []
conditions = []
for ticker in tickers:
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(df) < sequence_length:
            continue
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
        vix = vix.reindex(returns.index, method='ffill').values
        for i in range(0, len(returns) - sequence_length + 1, 10):
            seq = returns[i:i+sequence_length].values
            cond = vix[i:i+sequence_length]
            if len(seq) == sequence_length and len(cond) == sequence_length:
                sequences.append(seq)
                conditions.append(cond)
    except:
        continue

# Convert to tensor
sequences = torch.tensor(sequences, dtype=torch.float32)
conditions = torch.tensor(conditions, dtype=torch.float32)

# Standardize per sequence
sequences = (sequences - sequences.mean(dim=1, keepdim=True)) / sequences.std(dim=1, keepdim=True)
sequences = sequences * 0.015  # Scale to typical std
conditions = (conditions - conditions.mean()) / conditions.std()  # Normalize VIX

# Save
torch.save({"sequences": sequences, "conditions": conditions}, 
           f"{output_dir}/sequences_{sequence_length}.pt")
print(f"Saved {len(sequences)} sequences to sequences_{sequence_length}.pt")