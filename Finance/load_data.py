import yfinance as yf
import pandas as pd
import numpy as np
import torch
import os

# 配置参数
SEQ_LEN = 252
START_DATE = "2017-01-01"
END_DATE = "2024-12-31"
OUTPUT_DIR = "financial_data/sequences"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 获取S&P500成分股
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()

# 2. 下载历史数据
print("下载股票数据...")
data = yf.download(tickers, start=START_DATE, end=END_DATE, 
                  auto_adjust=True, group_by='ticker', progress=False)

# 3. 处理数据
sequences, date_feats, market_caps = [], [], []
for ticker in tickers:
    try:
        # 3.1 获取单只股票数据
        if ticker not in data.columns.levels[0]:
            continue
            
        closes = data[ticker]['Close'].dropna()
        if len(closes) < SEQ_LEN:
            continue
            
        # 3.2 计算对数收益率
        returns = np.log(closes / closes.shift(1)).dropna()
        
        # 3.3 按月截取序列
        monthly_starts = closes.groupby([closes.index.year, closes.index.month]).head(1).index
        for date in monthly_starts:
            idx = returns.index.get_loc(date)
            if idx + SEQ_LEN <= len(returns):
                seq = returns.iloc[idx:idx+SEQ_LEN].values
                
                # 3.4 生成特征
                date_feat = [
                    (date.year - 2017) / 8.0,  # 年份标准化
                    (date.month - 1) / 11.0,   # 月份标准化
                    (date.day - 1) / 30.0      # 日标准化
                ]
                
                # 3.5 获取市值
                ticker_data = yf.Ticker(ticker)
                market_cap = ticker_data.info.get('marketCap', 1e9)
                
                sequences.append(seq)
                date_feats.append(date_feat)
                market_caps.append(market_cap)
                
    except Exception as e:
        print(f"处理{ticker}时出错: {str(e)}")
        continue

# 4. 转换为张量
sequences = torch.tensor(np.array(sequences, dtype=np.float32))
date_feats = torch.tensor(np.array(date_feats, dtype=np.float32))
market_caps = torch.tensor(np.array(market_caps, dtype=np.float32)).unsqueeze(1)

# 5. 标准化市值
market_caps = (market_caps - market_caps.min()) / (market_caps.max() - market_caps.min() + 1e-8)

# 6. 保存数据
torch.save({
    "sequences": sequences,
    "start_dates": date_feats, 
    "market_caps": market_caps
}, os.path.join(OUTPUT_DIR, "sequences_252.pt"))

# 打印结果
print(f"\n最终数据尺寸:")
print(f"价格序列: {sequences.shape} (样本数×序列长度)")
print(f"日期特征: {date_feats.shape} (样本数×3)")
print(f"市值数据: {market_caps.shape} (样本数×1)")
print(f"\n示例统计量:")
print(f"收益率均值: {sequences.mean().item():.6f}")
print(f"收益率标准差: {sequences.std().item():.6f}")
print(f"市值范围: [{market_caps.min().item():.2f}, {market_caps.max().item():.2f}]")