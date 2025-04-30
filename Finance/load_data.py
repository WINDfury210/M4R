import yfinance as yf
import pandas as pd
import numpy as np
import torch
import os
import time

# 配置
sequence_length = 252
start_date = "2017-01-01"
end_date = "2024-12-31"
output_dir = "financial_data/sequences"
cache_dir = "financial_data/cache"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sequences_252.pt")
standardize = False  # 可调节标准化选项
target_std = 0.015   # 标准化时的目标标准差

# 获取 S&P 500 股票代码
try:
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
except Exception as e:
    print(f"Failed to fetch S&P 500 tickers: {e}")
    exit(1)
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()

# 下载和处理股票数据
sequences = []
start_dates = []
market_caps = []
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
        
        # 获取公司市值
        ticker_obj = yf.Ticker(ticker)
        market_cap = ticker_obj.info.get('marketCap', None)
        if market_cap is None:
            failed_tickers.append(ticker)
            print(f"Failed to get market cap for {ticker}")
            continue
        
        # 计算对数回报
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        if len(returns) < sequence_length:
            failed_tickers.append(ticker)
            continue
        
        # 按月截取序列
        df.index = pd.to_datetime(df.index)
        monthly_starts = df.groupby([df.index.year, df.index.month]).head(1).index
        for start_date in monthly_starts:
            try:
                start_idx = returns.index.get_loc(start_date)
                if start_idx + sequence_length <= len(returns):
                    seq = returns.iloc[start_idx:start_idx + sequence_length].values
                    if len(seq) == sequence_length and not np.any(np.isnan(seq)):
                        sequences.append(seq)
                        # 时间嵌入
                        date_vec = [
                            (start_date.year - 2017) / 8.0,  # 2017-2024
                            (start_date.month - 1) / 12.0,
                            (start_date.day - 1) / 31.0
                        ]
                        start_dates.append(date_vec)
                        # 市值嵌入（暂存原始市值）
                        market_caps.append(market_cap)
            except Exception as e:
                print(f"Skipping sequence for {ticker} at {start_date}: {e}")
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
market_caps = np.array(market_caps, dtype=np.float32)

# 归一化市值
if len(market_caps) > 0:
    market_caps = (market_caps - np.min(market_caps)) / (np.max(market_caps) - np.min(market_caps) + 1e-8)
    market_caps = market_caps.reshape(-1, 1)  # [num_sequences, 1]

# 可选标准化序列
if standardize:
    mean = np.mean(sequences, axis=1, keepdims=True)
    std = np.std(sequences, axis=1, keepdims=True)
    sequences = (sequences - mean) / (std + 1e-8) * target_std

# 转换为张量
sequences = torch.tensor(sequences)
start_dates = torch.tensor(start_dates)
market_caps = torch.tensor(market_caps)

# 维度检验
if sequences.shape[0] == 0:
    raise ValueError("No sequences generated")
if sequences.dim() != 2 or sequences.shape[1] != sequence_length:
    raise ValueError(f"Expected shape (N, {sequence_length}), got {sequences.shape}")
if start_dates.shape != (sequences.shape[0], 3):
    raise ValueError(f"Expected start_dates shape ({sequences.shape[0]}, 3), got {start_dates.shape}")
if market_caps.shape != (sequences.shape[0], 1):
    raise ValueError(f"Expected market_caps shape ({sequences.shape[0]}, 1), got {market_caps.shape}")
print(f"Sequences shape: {sequences.shape}")
print(f"Start dates shape: {start_dates.shape}")
print(f"Market caps shape: {market_caps.shape}")
print(f"Sequences mean: {sequences.mean().item():.6f}")
print(f"Sequences std: {sequences.std().item():.6f}")

# 保存
torch.save({
    "sequences": sequences,
    "start_dates": start_dates,
    "market_caps": market_caps
}, output_file)
print(f"Saved {sequences.shape[0]} sequences to {output_file}")