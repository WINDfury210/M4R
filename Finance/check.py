import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import ks_2samp, skew, kurtosis, wasserstein_distance
import os
import json
from tqdm import tqdm

# ==================== 配置参数 ====================
SEQUENCE_LENGTH = 252
NUM_SAMPLES = 1000
STEPS = 200
DATA_FILE = "financial_data/sequences/sequences_252.pt"
MODEL_FILE = "saved_models/financial_unet_final.pth"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 模型架构 ====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, time):
        emb = time[:, None] * self.emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class FinancialUNet(nn.Module):
    def __init__(self, seq_len=252, cond_dim=4, time_dim=128):
        super().__init__()
        self.seq_len = seq_len
        
        # 时间编码
        self.time_embed = TimeEmbedding(time_dim)
        
        # 条件编码 (3个日期特征 + 1个市值特征)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(time_dim, time_dim)
        )
        
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        # 中间层
        self.mid = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.up2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        
        # 输出层
        self.output = nn.Conv1d(32, 1, 1)
        
        # 条件注入层
        self.cond_inject = nn.Linear(time_dim, 32)

    def forward(self, x, t, cond):
        x = x.unsqueeze(1)  # [B, 1, seq_len]
        
        # 时间条件
        t_emb = self.time_embed(t)  # [B, time_dim]
        t_emb = self.cond_inject(t_emb).unsqueeze(-1)  # [B, 32, 1]
        
        # 项目条件
        c_emb = self.cond_proj(cond)  # [B, time_dim]
        c_emb = self.cond_inject(c_emb).unsqueeze(-1)  # [B, 32, 1]
        
        # 下采样
        x1 = self.down1(x) + t_emb + c_emb
        x2 = self.down2(F.max_pool1d(x1, 2)) + t_emb + c_emb
        
        # 中间层
        x_mid = self.mid(F.max_pool1d(x2, 2)) + t_emb + c_emb
        
        # 上采样
        x_up = F.interpolate(x_mid, scale_factor=2, mode='linear')
        x_up = self.up1(x_up) + F.interpolate(x2, scale_factor=2, mode='linear')
        
        x_up = F.interpolate(x_up, scale_factor=2, mode='linear')
        x_up = self.up2(x_up) + F.interpolate(x1, scale_factor=2, mode='linear')
        
        return self.output(x_up).squeeze(1)

# ==================== 扩散过程 ====================
class DiffusionProcess(nn.Module):
    def __init__(self, num_timesteps=200, device="cpu"):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 余弦调度
        t = torch.linspace(0, 1, num_timesteps+1)
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        betas = torch.clip(1 - (f[1:] / f[:-1]), 0, 0.999)
        
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1. - alpha_bars)
        
        # 注册缓冲区
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alpha_bars', alpha_bars.to(device))
        self.register_buffer('sqrt_alpha_bars', sqrt_alpha_bars.to(device))
        self.register_buffer('sqrt_one_minus_alpha_bars', sqrt_one_minus_alpha_bars.to(device))
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        noisy = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return noisy, noise
    
    def sample(self, model, cond, steps=100, eta=0.0):
        model.eval()
        with torch.no_grad():
            # 初始噪声
            x = torch.randn(cond.shape[0], self.seq_len, device=cond.device)
            
            # 时间步安排
            step_indices = torch.linspace(self.num_timesteps-1, 0, steps+1, dtype=torch.long)
            
            for i in tqdm(range(steps), desc="Sampling"):
                t = step_indices[i]
                t_next = step_indices[i+1] if i < steps-1 else -1
                
                # 预测噪声
                t_batch = torch.full((cond.shape[0],), t, device=cond.device)
                pred_noise = model(x, t_batch, cond)
                
                # DDIM采样
                alpha_bar = self.alpha_bars[t].view(-1, 1)
                alpha_bar_next = self.alpha_bars[t_next].view(-1, 1) if t_next >=0 else 1.0
                
                x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
                x = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1 - alpha_bar_next - eta**2) * pred_noise
                
                if eta > 0 and t_next >= 0:
                    x = x + eta * torch.sqrt(1 - alpha_bar_next) * torch.randn_like(x)
            
            return x

# ==================== 评估函数 ====================
def load_data():
    data = torch.load(DATA_FILE)
    sequences = data["sequences"].float()
    start_dates = data["start_dates"].float()
    market_caps = data["market_caps"].float()
    conditions = torch.cat([start_dates, market_caps], dim=1)
    return sequences, conditions

def calculate_metrics(real_data, generated_data):
    real_np = real_data.cpu().numpy()
    gen_np = generated_data.cpu().numpy()
    
    metrics = {
        # 基本统计量
        'real_mean': float(real_np.mean()),
        'gen_mean': float(gen_np.mean()),
        'real_std': float(real_np.std()),
        'gen_std': float(gen_np.std()),
        'real_skew': float(skew(real_np.flatten())),
        'gen_skew': float(skew(gen_np.flatten())),
        'real_kurtosis': float(kurtosis(real_np.flatten())),
        'gen_kurtosis': float(kurtosis(gen_np.flatten())),
        
        # 分布相似性
        'ks_stat': float(ks_2samp(real_np.flatten(), gen_np.flatten())[0]),
        'wasserstein': float(wasserstein_distance(real_np.flatten(), gen_np.flatten())),
        
        # 自相关比较
        'acf_real': acf(real_np[0], nlags=20).tolist(),
        'acf_gen': acf(gen_np[0], nlags=20).tolist(),
        'acf_mse': float(np.mean((acf(real_np[0], nlags=20) - acf(gen_np[0], nlags=20))**2)),
        
        # 波动率聚类
        'vol_real': calculate_volatility(real_np).tolist(),
        'vol_gen': calculate_volatility(gen_np).tolist(),
        'vol_corr': float(np.corrcoef(calculate_volatility(real_np), calculate_volatility(gen_np))[0,1])
    }
    return metrics

def calculate_volatility(data, window=20):
    """计算滚动波动率"""
    return np.array([np.std(data[:, i:i+window], axis=1).mean() 
                    for i in range(data.shape[1] - window)])

def visualize_results(real_data, generated_data, metrics, output_dir):
    # 样本路径可视化
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(generated_data[i], alpha=0.6, label=f'Generated {i+1}' if i == 0 else "")
        plt.plot(real_data[i], alpha=0.6, label=f'Real {i+1}' if i == 0 else "")
    plt.title("Sample Paths Comparison")
    plt.xlabel("Time")
    plt.ylabel("Returns")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "sample_paths.png"))
    plt.close()
    
    # 自相关函数比较
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['acf_real'], label='Real Data')
    plt.plot(metrics['acf_gen'], label='Generated Data')
    plt.title("Autocorrelation Function Comparison")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "acf_comparison.png"))
    plt.close()
    
    # 波动率比较
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['vol_real'], label='Real Data')
    plt.plot(metrics['vol_gen'], label='Generated Data')
    plt.title("Rolling Volatility Comparison (20-day window)")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "volatility_comparison.png"))
    plt.close()

# ==================== 主函数 ====================
def main():
    # 加载数据和模型
    sequences, conditions = load_data()
    model = FinancialUNet(seq_len=SEQUENCE_LENGTH, cond_dim=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    
    diffusion = DiffusionProcess(num_timesteps=200, device=DEVICE).to(DEVICE)
    
    # 生成样本
    print("Generating samples...")
    with torch.no_grad():
        generated_samples = diffusion.sample(model, conditions[:NUM_SAMPLES].to(DEVICE), steps=STEPS)
    
    # 计算评估指标
    print("Calculating metrics...")
    metrics = calculate_metrics(sequences[:NUM_SAMPLES], generated_samples)
    
    # 保存结果
    print("Saving results...")
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    torch.save(generated_samples, os.path.join(OUTPUT_DIR, "generated_samples.pt"))
    
    # 可视化
    visualize_results(sequences[:5].cpu(), generated_samples[:5].cpu(), metrics, OUTPUT_DIR)
    
    # 打印关键指标
    print("\nKey Metrics:")
    print(f"Mean - Real: {metrics['real_mean']:.6f}, Generated: {metrics['gen_mean']:.6f}")
    print(f"Std Dev - Real: {metrics['real_std']:.6f}, Generated: {metrics['gen_std']:.6f}")
    print(f"Skewness - Real: {metrics['real_skew']:.6f}, Generated: {metrics['gen_skew']:.6f}")
    print(f"Kurtosis - Real: {metrics['real_kurtosis']:.6f}, Generated: {metrics['gen_kurtosis']:.6f}")
    print(f"KS Statistic: {metrics['ks_stat']:.6f}")
    print(f"Wasserstein Distance: {metrics['wasserstein']:.6f}")
    print(f"Volatility Correlation: {metrics['vol_corr']:.6f}")
    print(f"ACF MSE: {metrics['acf_mse']:.6f}")
    
    print(f"\nResults saved to {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main()