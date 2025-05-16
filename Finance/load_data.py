# 终极S&P 500数据下载器 (v3.0)
import yfinance as yf
import pandas as pd
import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ================== 配置区 ==================
START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
MAX_RETRIES = 5                  # 单股票最大重试次数
INITIAL_DELAY = 3                # 初始延迟(秒)
MAX_DELAY = 30                   # 最大延迟(秒)
TIMEOUT = 15                     # 请求超时(秒)
BATCH_SIZE = 3                   # 每批请求数量
PROXY = None                     # 如需代理: {'http': 'http://your_proxy:port'}

# ================== 核心函数 ==================
def get_sp500_tickers():
    """获取最新成分股列表（带多重备用方案）"""
    backup_tickers = ['AAPL','MSFT','GOOG','AMZN','META','TSLA','NVDA','BRK-B','JPM','JNJ']
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10, proxies=PROXY)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        return [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
    except Exception as e:
        print(f"⚠️ 成分股获取失败: {e}, 使用备用列表")
        return backup_tickers

def download_with_strategy(tickers):
    """智能下载策略"""
    success = []
    failed = []
    
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        batch_results = {}
        
        # 批次下载
        for attempt in range(MAX_RETRIES):
            try:
                # 动态延迟（指数退避）
                delay = min(INITIAL_DELAY * (2 ​**​ attempt), MAX_DELAY) + random.uniform(0, 3)
                time.sleep(delay)
                
                print(f"🔄 尝试第{attempt+1}次: {batch}")
                data = yf.download(
                    batch,
                    start=START_DATE,
                    end=END_DATE,
                    group_by="ticker",
                    threads=True,
                    timeout=TIMEOUT,
                    proxy=PROXY
                )
                
                # 处理结果
                for ticker in batch:
                    if ticker in data:
                        df = data[ticker]
                        if not df.empty:
                            batch_results[ticker] = df
                            print(f"✅ {ticker} 下载成功 (数据量: {len(df)})")
                
                break  # 成功则跳出重试循环
            
            except Exception as e:
                print(f"❌ 批次失败: {type(e).__name__}: {str(e)[:100]}")
                if attempt == MAX_RETRIES - 1:
                    for ticker in batch:
                        if ticker not in batch_results:
                            failed.append(ticker)
                            print(f"🛑 最终失败: {ticker}")
        
        # 记录成功
        success.extend(batch_results.keys())
        
        # 保存批次数据
        if batch_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sp500_batch_{i//BATCH_SIZE}_{timestamp}.csv"
            pd.concat(batch_results, axis=0).to_csv(filename)
            print(f"💾 批次保存: {filename}")
    
    return success, failed

# ================== 主程序 ==================
if __name__ == "__main__":
    print("="*50)
    print("📈 S&P 500历史数据下载器 (终极版)")
    print("="*50)
    
    # 获取股票列表
    tickers = get_sp500_tickers()
    print(f"\n🔄 获取到 {len(tickers)} 只成分股，开始下载...")
    
    # 执行下载
    start_time = time.time()
    success, failed = download_with_strategy(tickers)
    
    # 结果统计
    print("\n" + "="*50)
    print(f"🎉 下载完成! 耗时: {time.time()-start_time:.1f}秒")
    print(f"✅ 成功: {len(success)} 只")
    print(f"❌ 失败: {len(failed)} 只")
    
    if failed:
        print("\n⚠️ 失败股票列表:")
        print(", ".join(failed))
        print("建议：单独重试失败股票或更换IP后重试")