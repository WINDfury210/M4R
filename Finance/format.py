import yfinance as yf
import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime
import time

# 配置
start_date = "2019-01-01"
end_date = "2024-12-31"
output_dir = "financial_data/sequences"
cache_dir = "financial_data/cache"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sequences_252.pt")
standardize = False  # 可调节标准化选项
target_std = 0.015   # 标准化时的目标标准差

# 获取 S&P 500 股票代码和行业分类
sp500 = pd.read_html('[invalid url, do not cite])[0]
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()[:400]
company_info = sp500.set_index('Symbol')['GICS Sector'].to_dict()
company_info = {k.replace('.', '-'): v for k, v in company_info.items()}

# 下载和处理股票数据
sequences = []
start_dates = []
sectors = []
failed_tickers = []

for ticker in tickers:
    cache_file = os.path.join(cache_dir, f"{ticker}.csv")
    try:
        # 从缓存加载或下载数据
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
        else:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df.empty or len(df) < sequence_length:
                failed_tickers.append(ticker)
                continue
            df.to_csv(cache_file, date_format='%Y-%m-%d')
        
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        if len(returns) < sequence_length:
            failed_tickers.append(ticker)
            continue
        
        # 按月截取序列
        df.index = pd.to_datetime(df.index)
        monthly_starts = df.groupby([df.index.year, df.index.month]).head(1).index
        for start_date in monthly_starts:
            start_idx = returns.index.get_loc(start_date)
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
        print(f"Processed {ticker}: {len(monthly_starts)} sequences")
    except Exception as e:
        failed_tickers.append(ticker)
        print(f"Failed {ticker}: {e}")
    time.sleep(1)  # 避免 API 限制

if failed_tickers:
    print(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")

# 转换为数组
sequences = np.array(sequences, dtype=np.float32)
start_dates = np.array(start_dates, dtype=np.float32)
sectors = np.array(sectors)

# 可选标准化
if standardize:
    mean = np.mean(sequences, axis=1, keepdims=True)
    std = np.std(sequences, axis=1, keepdims=True)
    sequences = (sequences - mean) / (std + 1e-8) * target_std

# 转换为张量
sequences = torch.tensor(sequences)
start_dates = torch.tensor(start_dates)

# 维度检验
if sequences.shape[0] == 0:
    raise ValueError("No sequences generated")
if sequences.dim() != 2 or sequences.shape[1] != sequence_length:
    raise ValueError(f"Expected shape (N, {sequence_length}), got {sequences.shape}")
if start_dates.shape != (sequences.shape[0], 3):
    raise ValueError(f"Expected start_dates shape ({sequences.shape[0]}, 3), got {start_dates.shape}")
if sectors.shape != (sequences.shape[0],):
    raise ValueError(f"Expected sectors shape ({sequences.shape[0]},), got {sectors.shape}")
print(f"Sequences shape: {sequences.shape}")
print(f"Start dates shape: {start_dates.shape}")
print(f"Sectors shape: {sectors.shape}")
print(f"Sequences mean: {sequences.mean().item():.6f}")
print(f"Sequences std: {sequences.std().item():.6f}")

# 保存
torch.save({
    "sequences": sequences,
    "start_dates": start_dates,
    "sectors": sectors
}, output_file)
print(f"Saved {sequences.shape[0]} sequences to {output_file}")