"""Diffusion Model Validation Script

This script validates a trained conditional diffusion model by:
1. Loading the trained model and original dataset
2. Generating samples under different conditions
3. Comparing statistics between real and generated samples
4. Saving results and visualizations
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


# 1. 扩散过程 ==========================================================
class DiffusionProcess:
    def __init__(self, num_timesteps: int = 1000, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 线性噪声调度
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

# 2. 时间嵌入 ==========================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        emb = time.float().view(-1, 1) * self.emb.view(1, -1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# 3. 条件UNet模型 ======================================================
class ConditionalUNet(nn.Module):
    def __init__(self, seq_len: int = 252):
        super().__init__()
        self.seq_len = seq_len
        
        # 时间/条件嵌入
        self.time_embed = TimeEmbedding(128)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, 128),  # date(3) + market_cap(1)
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # 输入层 [B,1,252]
        self.input_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # [B,64,126]
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # [B,128,63]
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        # 中间层 [B,128,63]
        self.mid_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B,64,126]
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B,32,252]
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        
        # 输出层 [B,32,252] -> [B,1,252]
        self.output_conv = nn.Conv1d(32, 1, kernel_size=1)
        
        # 条件注入层（每层独立）
        self.cond_inject_input = nn.Linear(128, 32)
        self.cond_inject_down1 = nn.Linear(128, 64)
        self.cond_inject_down2 = nn.Linear(128, 128)
        self.cond_inject_mid = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor, t: torch.Tensor, date: torch.Tensor, market_cap: torch.Tensor) -> torch.Tensor:
        # 条件嵌入
        t_emb = self.time_embed(t)  # [B,128]
        cond = torch.cat([date, market_cap], dim=-1)  # [B,4]
        c_emb = self.cond_proj(cond)  # [B,128]
        combined_cond = t_emb + c_emb  # [B,128]
        
        # 输入处理 [B,1,252]
        x = x.unsqueeze(1)  # 确保输入是3D张量
        h0 = self.input_conv(x)
        
        # 下采样路径 + 条件注入
        h0 = h0 + self.cond_inject_input(combined_cond).unsqueeze(-1)
        h1 = self.down1(h0)
        h1 = h1 + self.cond_inject_down1(combined_cond).unsqueeze(-1)
        h2 = self.down2(h1)
        h2 = h2 + self.cond_inject_down2(combined_cond).unsqueeze(-1)
        
        # 中间层
        h_mid = self.mid_conv(h2)
        h_mid = h_mid + self.cond_inject_mid(combined_cond).unsqueeze(-1)
        
        # 上采样路径 + 跳连
        h_up1 = self.up1(h_mid)
        h_up1 = h_up1 + h1  # 跳连
        h_up2 = self.up2(h_up1)
        h_up2 = h_up2 + h0  # 跳连
        
        # 输出处理
        out = self.output_conv(h_up2)
        return out.squeeze(1)  # [B,252]

# 4. 数据加载 ==========================================================
class FinancialDataset(Dataset):
    def __init__(self, data_path: str):
        data = torch.load(data_path)
        self.sequences = data["sequences"]      # [N,252]
        self.dates = data["start_dates"]        # [N,3]
        self.market_caps = data["market_caps"]  # [N,1]
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx],
            "market_cap": self.market_caps[idx]
        }
        
def load_model_and_data(model_path, data_path, device="cuda"):
    """Load trained model and original dataset."""
    # Load model
    model_info = torch.load(model_path, map_location=device)
    model = ConditionalUNet(seq_len=252).to(device)
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    
    # Load original data
    dataset = FinancialDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return model, dataloader


def generate_samples(model, diffusion, conditions, num_samples=100, steps=200, device="cuda"):
    """Generate samples using DDIM sampling."""
    with torch.no_grad():
        # Prepare conditions
        dates = conditions["date"].repeat(num_samples, 1).to(device)
        market_caps = conditions["market_cap"].repeat(num_samples, 1).to(device)
        
        # Initialize noise
        samples = torch.randn(num_samples, 252, device=device)
        
        # Generation loop
        for t in tqdm(reversed(range(0, diffusion.num_timesteps, diffusion.num_timesteps // steps)),
                     desc="Generating samples"):
            times = torch.full((num_samples,), t, device=device, dtype=torch.long)
            pred_noise = model(samples, times, dates, market_caps)
            
            # DDIM update rule
            alpha_bar = diffusion.alpha_bars[t]
            alpha_bar_prev = diffusion.alpha_bars[t-1] if t > 0 else torch.tensor(1.0)
            
            samples = (samples - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            samples = samples * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev).sqrt() * pred_noise
        
        return samples.cpu()


def calculate_statistics(real_data, generated_data):
    """Calculate statistics comparing real and generated data."""
    stats = {}
    
    # Basic statistics
    stats['real_mean'] = real_data.mean().item()
    stats['generated_mean'] = generated_data.mean().item()
    stats['real_std'] = real_data.std().item()
    stats['generated_std'] = generated_data.std().item()
    
    # Correlation statistics
    real_corr = np.corrcoef(real_data.numpy())
    generated_corr = np.corrcoef(generated_data.numpy())
    stats['real_corr_mean'] = np.abs(real_corr).mean()
    stats['generated_corr_mean'] = np.abs(generated_corr).mean()
    
    # Distribution similarity
    real_hist = np.histogram(real_data.numpy(), bins=50, density=True)[0]
    generated_hist = np.histogram(generated_data.numpy(), bins=50, density=True)[0]
    stats['hist_kl'] = F.kl_div(
        torch.from_numpy(generated_hist).log(), 
        torch.from_numpy(real_hist), 
        reduction='batchmean'
    ).item()
    
    return stats


def save_results(real_samples, generated_samples, stats, output_dir):
    """Save comparison visualizations and statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistics
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Sample comparison plot
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.plot(real_samples[i].numpy())
        plt.title(f"Real Sample {i+1}")
        
        plt.subplot(2, 3, i+4)
        plt.plot(generated_samples[i].numpy())
        plt.title(f"Generated Sample {i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_comparison.png'))
    plt.close()
    
    # Metrics comparison plot
    metrics = ['mean', 'std', 'corr_mean']
    plt.figure(figsize=(10, 4))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        plt.bar(['Real', 'Generated'], 
                [stats['all'][f'real_{metric}'], stats['all'][f'generated_{metric}']])
        plt.title(metric.upper())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()


def validate(model_path, data_path, num_test_conditions=5, samples_per_cond=100, output_dir="validation_results"):
    """Main validation workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize components
    diffusion = DiffusionProcess(device=device)
    model, dataloader = load_model_and_data(model_path, data_path, device)
    
    # Get test conditions
    all_data = next(iter(dataloader))
    cond_indices = torch.randperm(len(all_data["date"]))[:num_test_conditions]
    
    # Storage for results
    all_stats = {}
    real_samples = []
    generated_samples = []
    
    # Get global real data
    real_all = torch.cat([batch["sequence"] for batch in dataloader], dim=0)
    
    # Generate and compare samples
    for i, idx in enumerate(cond_indices):
        condition = {
            "date": all_data["date"][idx].unsqueeze(0),
            "market_cap": all_data["market_cap"][idx].unsqueeze(0)
        }
        
        # Generate samples
        gen_data = generate_samples(
            model, diffusion, condition, 
            num_samples=samples_per_cond,
            device=device
        )
        
        # Get corresponding real samples
        real_data = all_data["sequence"][idx].unsqueeze(0)
        
        # Calculate statistics
        cond_stats = calculate_statistics(real_data, gen_data)
        all_stats[f'condition_{i}'] = cond_stats
        
        # Store samples for visualization
        if i < 3:
            real_samples.append(real_data.squeeze())
            generated_samples.append(gen_data[0])
    
    # Calculate global statistics
    gen_all = torch.cat([
        generate_samples(
            model, diffusion, 
            {"date": all_data["date"][i].unsqueeze(0), 
             "market_cap": all_data["market_cap"][i].unsqueeze(0)},
            num_samples=samples_per_cond // num_test_conditions,
            device=device
        ) for i in cond_indices
    ])
    all_stats['all'] = calculate_statistics(real_all, gen_all)
    
    # Print key statistics
    print("\n=== Key Statistics ===")
    print(f"Real data mean: {all_stats['all']['real_mean']:.4f}")
    print(f"Generated data mean: {all_stats['all']['generated_mean']:.4f}")
    print(f"Real data std: {all_stats['all']['real_std']:.4f}")
    print(f"Generated data std: {all_stats['all']['generated_std']:.4f}")
    print(f"KL divergence: {all_stats['all']['hist_kl']:.4f}")
    
    # Save results
    save_results(real_samples, generated_samples, all_stats, output_dir)
    print(f"\nValidation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    validate(
        model_path="conditional_diffusion_final.pth",
        data_path="financial_data/sequences/sequences_252.pt",
        num_test_conditions=5,
        samples_per_cond=100,
        output_dir="validation_results"
    )