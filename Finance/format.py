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
            if df.empty or len(df) < sequence_length:
                raise ValueError("Insufficient data")
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
                df = pd.read_csv(cache_file, index_col=0, parse_dates=['Date'])
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                logging.info(f"Loaded {ticker}: {df.shape}")
            else:
                df = download_with_retry(ticker, start_date, end_date)
                df.to_csv(cache_file, date_format='%Y-%m-%d')
                logging.info(f"Downloaded {ticker}: {df.shape}")
            
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"Index for {ticker} is not DatetimeIndex")
            
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            if returns.empty or len(returns) < sequence_length:
                failed_tickers.append(ticker)
                logging.warning(f"Skipping {ticker}: Not enough returns")
                continue
            
            monthly_starts = df.groupby([df.index.year, df.index.month]).head(1).index
            for start_date in monthly_starts:
                try:
                    start_date = pd.Timestamp(start_date)
                    start_idx = returns.index.get_loc(start_date, method='nearest')
                    if start_idx + sequence_length <= len(returns):
                        seq = returns.iloc[start_idx:start_idx + sequence_length].values
                        if len(seq) == sequence_length and not np.any(np.isnan(seq)):
                            sequences.append(seq)
                            date_vec = [
                                (start_date.year - 2019) / 6.0,
                                (start_date.month - 1) / 12.0,
                                (start_date.day - 1) / 31.0
                            ]
                            start_dates.append(date_vec)
                            sectors.append(company_info.get(ticker, 'Unknown'))
                except Exception as e:
                    logging.warning(f"Skipping sequence for {ticker} at {start_date}: {e}")
            logging.info(f"Processed {ticker}: {len(monthly_starts)} sequences")
        except Exception as e:
            failed_tickers.append(ticker)
            logging.error(f"Failed {ticker}: {e}")
        time.sleep(10)

if failed_tickers:
    logging.info(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

# Convert to arrays
sequences = np.array(sequences)
start_dates = np.array(start_dates)
sectors = np.array(sectors)

# Convert to tensor
sequences = torch.tensor(sequences, dtype=torch.float32)
start_dates = torch.tensor(start_dates, dtype=torch.float32)

# Dimension check
if sequences.shape[0] == 0:
    raise ValueError("No sequences generated")
if sequences.dim() != 2 or sequences.shape[1] != sequence_length:
    raise ValueError(f"Expected shape (N, {sequence_length}), got {sequences.shape}")
if start_dates.shape != (sequences.shape[0], 3):
    raise ValueError(f"Expected start_dates shape ({sequences.shape[0]}, 3), got {start_dates.shape}")
if sectors.shape != (sequences.shape[0],):
    raise ValueError(f"Expected sectors shape ({sequences.shape[0]},), got {sectors.shape}")
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