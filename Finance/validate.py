"""
Diffusion Model Validation Script
Generate 10 sequences per year (2017-2024) with DDPM and compute per-sample metrics

### 计算指标清单

`calculate_metrics` 函数为每个样本组和全局数据集计算以下指标，用于评估生成序列与真实序列的相似性：

#### 全局统计（metrics['global']）
- **real_mean**: 真实序列的均值，反映数据中心位置。
- **gen_mean**: 生成序列的均值，与 real_mean 比较以评估偏差。
- **real_std**: 真实序列的标准差，衡量波动性（目标 ~0.036374）。
- **gen_std**: 生成序列的标准差，评估波动性匹配程度。
- **real_corr**: 真实序列的滞后一阶自相关系数，反映相邻点的相关性。
- **gen_corr**: 生成序列的滞后一阶自相关系数，评估时间依赖性。
- **real_acf**: 真实序列的平均自相关（滞后 1-20），衡量长期时间结构。
- **gen_acf**: 生成序列的平均自相关，评估时间模式匹配。
- **ks_stat**: Kolmogorov-Smirnov 统计量，衡量分布差异（目标 < 0.05）。
- **wass_dist**: Wasserstein 距离，评估分布的平滑差异。
- **shapiro_pval**: 生成序列的 Shapiro-Wilk 正态性检验 p 值，评估正态性。
- **real_skew**: 真实序列的偏度，衡量分布不对称性。
- **gen_skew**: 生成序列的偏度，评估不对称性匹配。
- **real_kurt**: 真实序列的峰度，衡量分布尾部形状。
- **gen_kurt**: 生成序列的峰度，评估尾部特征。

#### 每组样本统计（metrics['year_{year}']）
- 每组样本（每年 10 组）计算上述指标（除全局统计外），用于逐样本分析。
- 输出报告显示前 3 组样本的指标，包含均值、标准差、相关性、自相关、KS 统计量、Wasserstein 距离、偏度和峰度。
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy import stats
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from model import ConditionalUNet1D  # Import model

# 1. Diffusion Process -------------------------------------------------------

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self._sigmoid_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # Posterior variance for DDPM
        self.posterior_variance = torch.zeros_like(self.betas)
        self.posterior_variance[1:] = self.betas[1:] * (1. - self.alpha_bars[:-1]) / (1. - self.alpha_bars[1:])
        self.posterior_variance[0] = self.betas[0]
    
    def _sigmoid_beta_schedule(self):
        t = torch.linspace(0, 1, self.num_timesteps)
        s = 0.015
        betas = 0.0001 + (0.01 - 0.0001) * torch.sigmoid(10 * (t - s))
        return betas
    
    def add_noise(self, x0, t, year=None):
        noise_scale = 0.0005
        if year is not None:
            year_std = {2019: 0.05, 2020: 0.06, 2021: 0.055}.get(year, 0.036374) / 1.0
            noise_scale = 0.005 * (year_std / 0.06)
            noise_scale = min(noise_scale, 0.002)
        noise = torch.randn_like(x0, device=self.device) * noise_scale
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

# 2. Data Loading ------------------------------------------------------------

class FinancialDataset(Dataset):
    def __init__(self, data_path, scale_factor=1.0):
        data = torch.load(data_path)
        self.sequences = data["sequences"]
        self.dates = data["start_dates"]
        # Store scaling parameters
        self.original_mean = self.sequences.mean().item()
        self.original_std = self.sequences.std().item()
        self.scale_factor = scale_factor
        # Apply scaling
        self.sequences = (self.sequences - self.original_mean) / self.original_std * scale_factor
        print(f"Scaled data - Mean: {self.sequences.mean().item():.6f}, Std: {self.sequences.std().item():.6f}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx]
        }
    
    def get_annual_start_dates(self, years):
        min_year, max_year = 2017, 2024
        start_dates = []
        for year in years:
            norm_year = (year - min_year) / (max_year - min_year)
            start_date = torch.tensor([norm_year, 0.0, 0.0], dtype=torch.float32)
            start_dates.append(start_date)
        return torch.stack(start_dates)
    
    def get_real_sequences_for_year(self, annual_date, num_samples=10):
        date_diffs = torch.norm(self.dates - annual_date, dim=-1)
        closest_indices = torch.argsort(date_diffs)[:num_samples]
        return self.sequences[closest_indices], closest_indices
    
    def inverse_scale(self, sequences):
        """Reverse scaling to original data scale"""
        return sequences * self.original_std / self.scale_factor + self.original_mean

# 3. Validation Core ---------------------------------------------------------

@torch.no_grad()
def generate_samples(model, diffusion, condition, num_samples, device, steps=1000):
    """Generate samples with DDPM sampling"""
    model.eval()
    labels = condition["date"].repeat(num_samples, 1).to(device)
    year = int(condition["date"][0, 0].item() * (2024 - 2017) + 2017)
    x = torch.randn(num_samples, 256, device=device)
    for t in reversed(range(diffusion.num_timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, labels)
        # DDPM update
        alpha_t = diffusion.alphas[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = diffusion.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        posterior_variance_t = diffusion.posterior_variance[t].view(-1, 1)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / sqrt_one_minus_alpha_bar_t) * pred_noise
        ) + torch.sqrt(posterior_variance_t) * noise
        x = torch.clamp(x, -5, 5)
    print(f"Year {year}: Scaled Gen mean={x.mean().item():.6f}, std={x.std().item():.6f}")
    return x.cpu()

def calculate_metrics(real_data, generated_data):
    metrics = {}
    
    if real_data.dim() == 1:
        real_data = real_data.unsqueeze(0)
    if generated_data.dim() == 1:
        generated_data = generated_data.unsqueeze(0)
    
    # Basic statistics
    metrics['real_mean'] = real_data.mean().item()
    metrics['gen_mean'] = generated_data.mean().item()
    metrics['real_std'] = real_data.std().item()
    metrics['gen_std'] = generated_data.std().item()
    
    real_data_np = real_data.cpu().numpy().flatten()
    gen_data_np = generated_data.cpu().numpy().flatten()
    
    # Correlation
    real_lagged = real_data_np[:-1]
    real_next = real_data_np[1:]
    gen_lagged = gen_data_np[:-1]
    gen_next = gen_data_np[1:]
    metrics['real_corr'] = np.corrcoef(real_lagged, real_next)[0, 1] if len(real_lagged) > 1 else 0.0
    metrics['gen_corr'] = np.corrcoef(gen_lagged, gen_next)[0, 1] if len(gen_lagged) > 1 else 0.0
    
    # Autocorrelation
    real_acf = acf(real_data_np, nlags=20, fft=True)[1:].mean()
    gen_acf = acf(gen_data_np, nlags=20, fft=True)[1:].mean()
    metrics['real_acf'] = real_acf
    metrics['gen_acf'] = gen_acf
    
    # KS Statistic
    ks_stat, ks_pval = stats.ks_2samp(real_data_np, gen_data_np)
    metrics['ks_stat'] = ks_stat
    
    # Wasserstein Distance
    wass_dist = stats.wasserstein_distance(real_data_np, gen_data_np)
    metrics['wass_dist'] = wass_dist
    
    # Shapiro-Wilk Test
    _, shapiro_pval = stats.shapiro(gen_data_np[:5000])  # Limit for Shapiro
    metrics['shapiro_pval'] = shapiro_pval
    
    # Skewness and Kurtosis
    metrics['real_skew'] = stats.skew(real_data_np)
    metrics['gen_skew'] = stats.skew(gen_data_np)
    metrics['real_kurt'] = stats.kurtosis(real_data_np)
    metrics['gen_kurt'] = stats.kurtosis(gen_data_np)
    
    return metrics

def print_enhanced_report(metrics_dict, years):
    print("\n=== Validation Report ===")
    
    # Global Statistics
    print("\n[Global Statistics]")
    print(f"{'Metric':<12} {'Real':>12} {'Generated':>12}")
    print("-" * 36)
    global_stats = metrics_dict['global']
    print(f"{'Mean':<12} {global_stats['real_mean']:>12.6f} {global_stats['gen_mean']:>12.6f}")
    print(f"{'Std Dev':<12} {global_stats['real_std']:>12.6f} {global_stats['gen_std']:>12.6f}")
    print(f"{'Corr':<12} {global_stats['real_corr']:>12.6f} {global_stats['gen_corr']:>12.6f}")
    print(f"{'Autocorr':<12} {global_stats['real_acf']:>12.6f} {global_stats['gen_acf']:>12.6f}")
    print(f"{'KS Stat':<12} {'-':>12} {global_stats['ks_stat']:>12.6f}")
    print(f"{'Wass Dist':<12} {'-':>12} {global_stats['wass_dist']:>12.6f}")
    print(f"{'Shapiro P':<12} {'-':>12} {global_stats['shapiro_pval']:>12.6f}")
    print(f"{'Skew':<12} {global_stats['real_skew']:>12.6f} {global_stats['gen_skew']:>12.6f}")
    print(f"{'Kurtosis':<12} {global_stats['real_kurt']:>12.6f} {global_stats['gen_kurt']:>12.6f}")
    
    # Per-sample statistics (first 3 samples per year)
    print("\n[Per-Sample Statistics]")
    print(f"{'Year':<6} {'Sample':<8} {'Metric':<12} {'Real':>12} {'Generated':>12}")
    print("-" * 50)
    for year in years:
        for i, sample_metrics in enumerate(metrics_dict[f'year_{year}'][:3]):
            for metric in ['mean', 'std', 'corr', 'acf', 'ks_stat', 'wass_dist', 'skew', 'kurt']:
                if metric in ['ks_stat', 'wass_dist', 'shapiro_pval']:
                    real_val = '-'
                    gen_val = f"{sample_metrics[metric]:.6f}"
                else:
                    real_key = f'real_{metric}'
                    gen_key = f'gen_{metric}'
                    real_val = f"{sample_metrics[real_key]:.6f}"
                    gen_val = f"{sample_metrics[gen_key]:.6f}"
                print(f"{year:<6} {i+1:<8} {metric.capitalize():<12} {real_val:>12} {gen_val:>12}")

def save_visualizations(real_samples, gen_samples, metrics, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'metrics_{year}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    plt.figure(figsize=(15, 10))
    for i in range(min(3, len(real_samples))):
        # Time series plots
        plt.subplot(3, 3, i+1)
        plt.plot(real_samples[i].numpy(), label="Real")
        plt.title(f"Real Sample {i+1} (Year {year})")
        plt.legend()
        plt.subplot(3, 3, i+4)
        plt.plot(gen_samples[i].numpy(), label="Generated")
        plt.title(f"Generated Sample {i+1} (Year {year})")
        plt.legend()
        # Q-Q Plot
        plt.subplot(3, 3, i+7)
        stats.probplot(gen_samples[i].numpy(), dist="norm", plot=plt)
        plt.title(f"Q-Q Plot Gen Sample {i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_samples.png'))
    plt.close()

# 5. Main Validation Function ------------------------------------------------

def run_validation(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DiffusionProcess(device=device, num_timesteps=1000)
    model = ConditionalUNet1D(seq_len=256, channels=config["channels"]).to(device)
    checkpoint = torch.load(config["model_path"], map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    dataset = FinancialDataset(config["data_path"], scale_factor=1.0)  # Same scaling as training
    years = list(range(2017, 2025))
    num_groups_per_year = 10
    annual_dates = dataset.get_annual_start_dates(years).to(device)
    metrics = {}
    all_gen_samples = []
    all_real_samples = []
    
    for year in years:
        year_idx = year - 2017  # Compute index for annual_dates
        annual_date = annual_dates[year_idx].to(device)
        condition = {"date": annual_date.unsqueeze(0)}
        real_sequences, _ = dataset.get_real_sequences_for_year(
            annual_date.cpu(), num_samples=num_groups_per_year)
        year_metrics_list = []
        year_gen_samples = []
        year_real_samples = []  # Initialize list
        gen_data = generate_samples(
            model, diffusion, condition,
            num_samples=num_groups_per_year,
            device=device,
            steps=2000
        )
        # Inverse scale generated and real data
        gen_data = dataset.inverse_scale(gen_data)
        real_sequences = dataset.inverse_scale(real_sequences)
        for group_idx in range(num_groups_per_year):
            group_data = gen_data[group_idx:group_idx+1]
            real_data = real_sequences[group_idx].unsqueeze(0)
            group_metrics = calculate_metrics(real_data, group_data)
            year_metrics_list.append(group_metrics)
            if group_idx < 3:
                year_real_samples.append(real_data.squeeze())
                year_gen_samples.append(group_data.squeeze())
            all_gen_samples.append(group_data.squeeze())
            all_real_samples.append(real_data.squeeze())
        metrics[f'year_{year}'] = year_metrics_list
        save_visualizations(year_real_samples, year_gen_samples, year_metrics_list, year, config["save_dir"])
    
    gen_all = torch.cat(all_gen_samples, dim=0)
    real_all_subset = torch.cat(all_real_samples, dim=0)
    metrics['global'] = calculate_metrics(real_all_subset, gen_all)
    print_enhanced_report(metrics, years)
    print(f"\nValidation complete! Results saved to {config['save_dir']}")

if __name__ == "__main__":
    config = {
        "model_path": "saved_models/final_model.pth",
        "data_path": "financial_data/sequences/sequences_256.pt",
        "save_dir": "validation_results",
        "channels": [32, 128, 256, 512, 1024, 2048]
    }
    os.makedirs(config["save_dir"], exist_ok=True)
    run_validation(config)