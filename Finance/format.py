import yfinance as yf
import pandas as pd
import torch
import numpy as np
import os
import time
import logging
logging.basicConfig(level=logging.DEBUG)

# Config
sequence_length = 252
start_date = "2019-01-01"
end_date = "2024-12-31"
output_dir = "financial_data/sequences"
cache_dir = "financial_data/cache"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Load tickers and sectors
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()[:400]
company_info = sp500.set_index('Symbol')['GICS Sector'].to_dict()
company_info = {k.replace('.', '-'): v for k, v in company_info.items()}

# Download with retry
def download_with_retry(ticker, start, end, retries=10):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            return df
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(15)
    raise Exception(f"Failed to download {ticker} after {retries} attempts")

# Download data
sequences = []
start_dates = []
sectors = []
failed_tickers = []
batch_size = 3

for i in range(0, len(tickers), batch_size):
    for ticker in tickers[i:i + batch_size]:
        cache_file = os.path.join(cache_dir, f"{ticker}.csv")
        try:
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                logging.info(f"Loaded {ticker}: {df.shape}")
            else:
                df = download_with_retry(ticker, start_date, end_date)
                if df.empty or len(df) < sequence_length:
                    failed_tickers.append(ticker)
                    logging.warning(f"Skipping {ticker}: Insufficient data")
                    continue
                df.to_csv(cache_file)
                logging.info(f"Downloaded {ticker}: {df.shape}")
            
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            df.index = pd.to_datetime(df.index)
            monthly_starts = df.groupby([df.index.year, df.index.month]).head(1).index
            for start_date in monthly_starts:
                start_idx = returns.index.get_loc(start_date)
                if start_idx + sequence_length <= len(returns):
                    seq = returns.iloc[start_idx:start_idx + sequence_length].values
                    if len(seq) == sequence_length:
                        sequences.append(seq)
                        start_dates.append(start_date.strftime('%Y-%m-%d'))
                        sectors.append(company_info.get(ticker, 'Unknown'))
            logging.info(f"Processed {ticker}: {len(monthly_starts)} sequences")
        except Exception as e:
            failed_tickers.append(ticker)
            logging.error(f"Failed {ticker}: {e}")
        time.sleep(10)

if failed_tickers:
    logging.info(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

# Convert to arrays
sequences = np.squeeze(np.array(sequences))
start_dates = np.array(start_dates)
sectors = np.array(sectors)

# Convert to tensor
sequences = torch.tensor(sequences, dtype=torch.float32)

# Dimension check
if sequences.shape[0] == 0:
    raise ValueError("No sequences generated")
if sequences.dim() != 2 or sequences.shape[1] != sequence_length:
    raise ValueError(f"Expected shape (N, {sequence_length}), got {sequences.shape}")
logging.info(f"Sequences shape: {sequences.shape}")
logging.info(f"Start dates shape: {start_dates.shape}")
logging.info(f"Sectors shape: {sectors.shape}")
logging.info(f"Sequences mean: {sequences.mean().item():.6f}")
logging.info(f"Sequences std: {sequences.std().item():.6f}")

# Save
output_file = f"{output_dir}/sequences_{sequence_length}.pt"
torch.save({
    "sequences": sequences,
    "start_dates": start_dates,
    "sectors": sectors
}, output_file)
logging.info(f"Saved {len(sequences)} sequences to {output_file}")