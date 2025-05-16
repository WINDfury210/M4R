# 极简加强版S&P 500数据下载器
import yfinance as yf
import pandas as pd
import time
import random

# 1. 获取S&P 500成分股列表（带备用方案）
def get_sp500_tickers():
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].tolist()
    except:
        print("⚠️ 无法获取最新成分股，使用预设列表")
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V']

# 2. 下载函数（智能重试+随机延迟）
def download_data(ticker):
    for attempt in range(3):  # 最多重试3次
        try:
            # 随机延迟（指数退避）
            delay = random.uniform(1, 5) * (attempt + 1)
            time.sleep(delay)
            
            data = yf.download(
                ticker, 
                start="2010-01-01", 
                progress=False,
                timeout=10  # 设置超时
            )
            
            if not data.empty:
                data['Ticker'] = ticker
                return data
                
        except Exception as e:
            print(f"❌ {ticker} 第{attempt+1}次尝试失败: {str(e)[:50]}...")
    
    return pd.DataFrame()

# 3. 主程序
if __name__ == "__main__":
    print("🚀 开始下载S&P 500历史数据...")
    
    tickers = get_sp500_tickers()
    print(f"📊 共获取到 {len(tickers)} 只股票")
    
    all_data = []
    for i, ticker in enumerate(tickers, 1):
        print(f"\n📡 正在处理 {i}/{len(tickers)}: {ticker}")
        df = download_data(ticker)
        
        if not df.empty:
            all_data.append(df)
            print(f"✅ 成功下载 {ticker} (最近数据: {df.index[-1].date()})")
        else:
            print(f"🛑 最终失败: {ticker}")
        
        # 批次间延迟（1-3秒随机）
        time.sleep(random.uniform(1, 3))
    
    # 合并保存
    if all_data:
        result = pd.concat(all_data)
        result.to_csv("sp500_data_enhanced.csv")
        print(f"\n🎉 完成！已保存 {len(all_data)}/{len(tickers)} 只股票数据")
        print(f"📂 文件: sp500_data_enhanced.csv (共{len(result)}行)")
    else:
        print("所有股票下载失败，请检查网络或稍后再试")