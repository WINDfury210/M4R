# 极简版S&P 500数据下载器
import yfinance as yf
import pandas as pd
import time

# 1. 获取S&P 500成分股列表（精简版）
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return table['Symbol'].tolist()

# 2. 下载函数（自动重试）
def download_data(ticker):
    for _ in range(3):  # 最多重试3次
        try:
            data = yf.download(ticker, start="2010-01-01", progress=False)
            if not data.empty:
                data['Ticker'] = ticker  # 添加股票代码列
                return data
        except:
            time.sleep(2)  # 失败后等待2秒
    return pd.DataFrame()

# 3. 主程序
if __name__ == "__main__":
    print("开始下载S&P 500历史数据...")
    
    # 获取股票列表（如果失败使用默认的10只大盘股）
    try:
        tickers = get_sp500_tickers()
    except:
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V']
    
    # 逐只下载
    all_data = []
    for ticker in tickers:
        df = download_data(ticker)
        if not df.empty:
            all_data.append(df)
            print(f"已下载: {ticker} | 数据量: {len(df)}行")
        else:
            print(f"下载失败: {ticker}")
        time.sleep(1)  # 每只股票间隔1秒
    
    # 合并保存
    if all_data:
        result = pd.concat(all_data)
        result.to_csv("sp500_data.csv")
        print(f"\n完成！已保存到 sp500_data.csv (共{len(result)}行数据)")
    else:
        print("所有股票下载失败，请检查网络连接")