import yfinance as yf
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm

# 配置参数
SEQ_LEN = 252
START_DATE = "2017-01-01"
END_DATE = "2024-12-31"
OUTPUT_DIR = "financial_data/sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 获取S&P500成分股（带重试逻辑）
def get_sp500_tickers(retries=3):
    for i in range(retries):
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
        except Exception as e:
            print(f"获取成分股失败 (尝试 {i+1}/{retries}): {e}")
            if i == retries - 1:
                raise
    return []

tickers = get_sp500_tickers()
print(f"获取到{len(tickers)}只成分股")

# 2. 下载历史数据（分批处理）
def download_batch(tickers, batch_size=100):
    results = {}
    for i in tqdm(range(0, len(tickers), batch_size), desc="下载数据"):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, start=START_DATE, end=END_DATE, 
                             group_by='ticker', progress=False)
            for ticker in batch:
                if ticker in data.columns.levels[0]:
                    results[ticker] = data[ticker]
        except Exception as e:
            print(f"批次{i}-{i+batch_size}下载失败: {e}")
    return results

print("开始下载股票数据...")
data = download_batch(tickers)
valid_tickers = [t for t in tickers if t in data]
print(f"成功下载{len(valid_tickers)}只股票数据")

# 3. 处理数据（带空值检查）
sequences, date_feats, market_caps = [], [], []
for ticker in tqdm(valid_tickers, desc="处理数据"):
    try:
        closes = data[ticker]['Close'].dropna()
        if len(closes) < SEQ_LEN:
            continue
            
        # 计算对数收益率
        returns = np.log(closes / closes.shift(1)).dropna()
        
        # 按月截取序列
        monthly_starts = closes.groupby([closes.index.year, closes.index.month]).head(1).index
        for date in monthly_starts:
            try:
                idx = returns.index.get_loc(date)
                if idx + SEQ_LEN <= len(returns):
                    seq = returns.iloc[idx:idx+SEQ_LEN].values
                    if len(seq) == SEQ_LEN and not np.any(np.isnan(seq)):
                        # 日期特征
                        date_feat = [
                            (date.year - 2017) / 8.0,
                            (date.month - 1) / 11.0,
                            (date.day - 1) / 30.0
                        ]
                        
                        # 获取市值（带默认值）
                        try:
                            market_cap = yf.Ticker(ticker).info.get('marketCap', 1e9)
                        except:
                            market_cap = 1e9
                            
                        sequences.append(seq)
                        date_feats.append(date_feat)
                        market_caps.append(market_cap)
            except:
                continue
    except Exception as e:
        print(f"处理{ticker}时出错: {e}")
        continue

# 4. 转换为张量（带空值检查）
if not sequences:
    raise ValueError("没有生成有效数据序列")

sequences = torch.tensor(np.array(sequences, dtype=np.float32))
date_feats = torch.tensor(np.array(date_feats, dtype=np.float32))
market_caps = torch.tensor(np.array(market_caps, dtype=np.float32)).unsqueeze(1)

# 5. 标准化市值（带非空检查）
if len(market_caps) > 0:
    market_caps = (market_caps - market_caps.min(dim=0)[0]) / \
                 (market_caps.max(dim=0)[0] - market_caps.min(dim=0)[0] + 1e-8)

# 6. 保存数据
output_path = os.path.join(OUTPUT_DIR, "sequences_252.pt")
torch.save({
    "sequences": sequences,
    "start_dates": date_feats, 
    "market_caps": market_caps
}, output_path)

# 打印结果
print(f"\n成功处理数据:")
print(f"价格序列: {sequences.shape} (样本数×序列长度)")
print(f"日期特征: {date_feats.shape} (样本数×3)")
print(f"市值数据: {market_caps.shape} (样本数×1)")
print(f"\n关键统计量:")
print(f"收益率均值: {sequences.mean().item():.6f}")
print(f"收益率标准差: {sequences.std().item():.6f}")
if len(market_caps) > 0:
    print(f"市值范围: [{market_caps.min().item():.2f}, {market_caps.max().item():.2f}]")
print(f"\n数据已保存到: {output_path}")