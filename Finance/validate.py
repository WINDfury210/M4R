"""
Diffusion Model Validation Script
Generate 10 sequences per year (2017-2024) and compute metrics
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# 1. Model Definitions --------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim, embedding_type="sinusoidal", hidden_dim=1024):
        super().__init__()
        self.dim = dim
        self.embedding_type = embedding_type
        if embedding_type == "sinusoidal":
            half_dim = dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            self.register_buffer("emb", emb)
        elif embedding_type == "linear":
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )

    def forward(self, time):
        if self.embedding_type == "sinusoidal":
            emb = time[:, None] * self.emb[None, :]
            emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1, 0, 0))
            return emb
        elif self.embedding_type == "linear":
            time = time.unsqueeze(-1).float()
            return self.mlp(time)

class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, ch, seq_len = x.size()
        q = self.query(x).view(batch, -1, seq_len).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, seq_len)
        v = self.value(x).view(batch, -1, seq_len)
        attn = self.softmax(torch.bmm(q, k) / (ch // 8) ** 0.5)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch, ch, seq_len)
        return x + self.gamma * out

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ConditionalUNet1D(nn.Module):
    def __init__(self, seq_len=256, channels=[32, 64, 128, 256]):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_levels = len(channels)
        
        # Time embedding (output channels[-1] = 256)
        self.time_embed = TimeEmbedding(dim=channels[-1], embedding_type="sinusoidal")
        
        # Date embedding (6D periodic encoding -> channels[-1] = 256)
        self.date_embed = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, channels[-1]),
            nn.LayerNorm(channels[-1])
        )
        
        # Input layer
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_convs = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_channels = channels[0]
        for i, out_channels in enumerate(channels):
            self.encoder_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                              stride=2 if i>0 else 1, padding=1))
            self.encoder_res.append(ResidualBlock1D(out_channels, out_channels))
            self.attentions.append(SelfAttention1D(out_channels) if i in [1, 2, 3] else nn.Identity())
            in_channels = out_channels
        
        # Middle layer
        self.mid_conv1 = ResidualBlock1D(channels[-1], channels[-1])
        self.mid_conv2 = ResidualBlock1D(channels[-1], channels[-1])
        
        # Decoder
        self.decoder_convs = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        for i in range(len(channels)-1):
            in_channels = channels[-1-i] + channels[-2-i]
            out_channels = channels[-2-i]
            self.decoder_convs.append(nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ))
            self.decoder_res.append(ResidualBlock1D(out_channels, out_channels))
        
        # Output layer
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(channels[0], channels[0],
                              kernel_size=3, stride=2,
                              padding=1, output_padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU()
        )
        self.final_conv = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x, t, date):
        # Date periodic encoding
        year, month, day = date[:, 0], date[:, 1], date[:, 2]
        year_sin = torch.sin(2 * math.pi * year)
        year_cos = torch.cos(2 * math.pi * year)
        month_sin = torch.sin(2 * math.pi * month)
        month_cos = torch.cos(2 * math.pi * month)
        day_sin = torch.sin(2 * math.pi * day)
        day_cos = torch.cos(2 * math.pi * day)
        date_emb = torch.stack([year_sin, year_cos, month_sin, month_cos, day_sin, day_cos], dim=-1)
        
        # Condition embeddings
        time_emb = self.time_embed(t)
        date_emb = self.date_embed(date_emb)
        combined_cond = time_emb + date_emb
        
        # Input processing
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        skips = []
        
        # Encoder
        for i, (conv, res, attn) in enumerate(zip(self.encoder_convs, self.encoder_res, self.attentions)):
            x = F.relu(conv(x))
            x = res(x)
            x = attn(x)
            skips.append(x)
        
        # Middle layer + condition injection
        x = self.mid_conv1(x)
        cond = combined_cond.unsqueeze(-1)
        x = x + cond
        x = self.mid_conv2(x)
        
        # Decoder
        for i, (conv, res) in enumerate(zip(self.decoder_convs, self.decoder_res)):
            skip = skips[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = F.relu(conv(x))
            x = res(x)
        
        # Final output
        x = self.final_upsample(x)
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        return self.final_conv(x).squeeze(1)

# 2. Diffusion Process -------------------------------------------------------

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = self._cosine_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def _cosine_beta_schedule(self, s=0.008):
        steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
        f = torch.cos(((steps / self.num_timesteps + s) / (1 + s)) * math.pi / 2) ** 2
        return torch.clip(1 - f[1:] / f[:-1], 0, 0.999)
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

# 3. Data Loading ------------------------------------------------------------

class FinancialDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.sequences = data["sequences"]
        self.dates = data["start_dates"]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx]
        }
    
    def get_annual_start_dates(self, years):
        """Generate normalized start dates for given years (Jan 1)"""
        min_year, max_year = 2017, 2024
        start_dates = []
        for year in years:
            # Normalize year to [0, 1] based on 2017-2024
            norm_year = (year - min_year) / (max_year - min_year)
            start_date = torch.tensor([norm_year, 0.0, 0.0], dtype=torch.float32)
            start_dates.append(start_date)
        return torch.stack(start_dates)  # [num_years, 3]
    
    def get_real_sequences_for_year(self, annual_date, num_samples=10):
        """Find num_samples real sequences closest to annual_date"""
        date_diffs = torch.norm(self.dates - annual_date, dim=1)
        closest_indices = torch.argsort(date_diffs)[:num_samples]
        return self.sequences[closest_indices], closest_indices

# 4. Validation Core ---------------------------------------------------------

def generate_samples(model, diffusion, conditions, num_samples=1, steps=50, device="cuda"):
    """Generate samples with DDIM sampling"""
    with torch.no_grad():
        dates = conditions["date"].repeat(num_samples, 1).to(device)
        samples = torch.randn(num_samples, 256, device=device) * 0.05
        skip = diffusion.num_timesteps // steps
        for t in tqdm(reversed(range(0, diffusion.num_timesteps, skip)),
                     desc="Generating samples"):
            times = torch.full((num_samples,), t, device=device, dtype=torch.long)
            pred_noise = model(samples, times, dates)
            alpha_bar = diffusion.alpha_bars[t]
            alpha_bar_prev = diffusion.alpha_bars[max(t-skip, 0)]
            x0_pred = (samples - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            samples = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev).sqrt() * pred_noise
        samples = torch.clamp(samples, min=-3, max=3)
        return samples.cpu()

def print_enhanced_report(metrics_dict, years):
    """Enhanced report with per-year average metrics"""
    print("\n=== Enhanced Validation Report ===")
    
    # Global Statistics
    print("\n[Global Statistics]")
    print(f"{'Metric':<15} | {'Real':>12} | {'Generated':>12} | {'Difference':>12} | {'Z-score':>10}")
    print("-" * 70)
    global_stats = metrics_dict['global']
    z_score_mean = (global_stats['gen_mean'] - global_stats['real_mean']) / global_stats['real_std']
    z_score_std = (global_stats['gen_std'] - global_stats['real_std']) / global_stats['real_std']
    
    print(f"{'Mean':<15} | {global_stats['real_mean']:>12.6f} | {global_stats['gen_mean']:>12.6f} | "
          f"{abs(global_stats['gen_mean']-global_stats['real_mean']):>12.6f} | {z_score_mean:>10.2f}")
    print(f"{'Std Dev':<15} | {global_stats['real_std']:>12.6f} | {global_stats['gen_std']:>12.6f} | "
          f"{abs(global_stats['gen_std']-global_stats['real_std']):>12.6f} | {z_score_std:>10.2f}")
    print(f"{'Correlation':<15} | {global_stats['real_corr']:>12.6f} | {global_stats['gen_corr']:>12.6f} | "
          f"{abs(global_stats['gen_corr']-global_stats['real_corr']):>12.6f} | {'-':>10}")
    print(f"{'Autocorr':<15} | {global_stats['real_acf']:>12.6f} | {global_stats['gen_acf']:>12.6f} | "
          f"{abs(global_stats['gen_acf']-global_stats['real_acf']):>12.6f} | {'-':>10}")
    
    # Per-Year Statistics (average of 10 groups)
    print("\n[Per-Year Statistics (Average of 10 Groups)]")
    for year in years:
        year_metrics = metrics_dict[f'year_{year}']
        z_mean = (year_metrics['gen_mean'] - year_metrics['real_mean']) / year_metrics['real_std']
        z_std = (year_metrics['gen_std'] - year_metrics['real_std']) / year_metrics['real_std']
        
        print(f"\nYear {year}:")
        print(f"{'Metric':<15} | {'Real':>12} | {'Generated':>12} | {'Diff %':>11} | {'Z-score':>10}")
        print("-" * 70)
        print(f"{'Mean':<15} | {year_metrics['real_mean']:>12.6f} | {year_metrics['gen_mean']:>12.6f} | "
              f"{abs(year_metrics['gen_mean']-year_metrics['real_mean'])/year_metrics['real_mean']*100:>11.2f}% | {z_mean:>10.2f}")
        print(f"{'Std Dev':<15} | {year_metrics['real_std']:>12.6f} | {year_metrics['gen_std']:>12.6f} | "
              f"{abs(year_metrics['gen_std']-year_metrics['real_std'])/year_metrics['real_std']*100:>11.2f}% | {z_std:>10.2f}")
        print(f"{'Correlation':<15} | {year_metrics['real_corr']:>12.6f} | {year_metrics['gen_corr']:>12.6f} | {'-':>11} | {'-':>10}")
        print(f"{'Autocorr':<15} | {year_metrics['real_acf']:>12.6f} | {year_metrics['gen_acf']:>12.6f} | {'-':>11} | {'-':>10}")
    
    # Issue Diagnostics
    print("\n[Issue Diagnostics]")
    if global_stats['gen_std'] > global_stats['real_std'] * 1.5:
        print("⚠️ Excessive global volatility: Check noise schedule or reduce initial noise scale")
    if abs(z_score_mean) > 3:
        print("⚠️ Significant global mean shift: Check condition injection or model bias")
    if global_stats['gen_corr'] < global_stats['real_corr'] * 0.5:
        print("⚠️ Low global correlation: Consider increasing model capacity")
    if abs(global_stats['gen_acf'] - global_stats['real_acf']) > 0.1:
        print("⚠️ Global autocorrelation mismatch: Check temporal modeling")

def calculate_metrics(real_data, generated_data):
    """Calculate metrics for single or multiple sequences"""
    from statsmodels.tsa.stattools import acf
    metrics = {}
    
    # Ensure input shapes: [N, 256] or [1, 256]
    if real_data.dim() == 1:
        real_data = real_data.unsqueeze(0)
    if generated_data.dim() == 1:
        generated_data = generated_data.unsqueeze(0)
    
    # Basic statistics
    metrics['real_mean'] = real_data.mean().item()
    metrics['gen_mean'] = generated_data.mean().item()
    metrics['real_std'] = real_data.std().item()
    metrics['gen_std'] = generated_data.std().item()
    
    # Correlation (handle single sequence)
    real_data_np = real_data.numpy()
    gen_data_np = generated_data.numpy()
    if real_data_np.shape[0] > 1:
        real_corr = np.corrcoef(real_data_np, rowvar=False)
        gen_corr = np.corrcoef(gen_data_np, rowvar=False)
        metrics['real_corr'] = np.abs(real_corr).mean()
        metrics['gen_corr'] = np.abs(gen_corr).mean()
    else:
        metrics['real_corr'] = 0.0
        metrics['gen_corr'] = 0.0
    
    # Autocorrelation (lags 1 to 20)
    real_acf = acf(real_data_np.flatten(), nlags=20, fft=True)[1:].mean()
    gen_acf = acf(gen_data_np.flatten(), nlags=20, fft=True)[1:].mean()
    metrics['real_acf'] = real_acf
    metrics['gen_acf'] = gen_acf
    
    # Distribution similarity
    real_hist = np.histogram(real_data_np.flatten(), bins=50, density=True)[0]
    gen_hist = np.histogram(gen_data_np.flatten(), bins=50, density=True)[0]
    metrics['kl_div'] = F.kl_div(
        torch.from_numpy(gen_hist + 1e-10).log(),
        torch.from_numpy(real_hist + 1e-10),
        reduction='batchmean'
    ).item()
    
    return metrics

def save_visualizations(real_samples, gen_samples, metrics, year, output_dir):
    """Save visualizations and metrics for a specific year"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, f'metrics_{year}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Sample comparison plot (first 3 groups)
    plt.figure(figsize=(15, 5))
    for i in range(min(3, len(real_samples))):
        plt.subplot(2, 3, i+1)
        plt.plot(real_samples[i].numpy())
        plt.title(f"Real Sample {i+1} (Year {year})")
        
        plt.subplot(2, 3, i+4)
        plt.plot(gen_samples[i].numpy())
        plt.title(f"Generated Sample {i+1} (Year {year})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_samples.png'))
    plt.close()

# 5. Main Validation Function ------------------------------------------------

def run_validation(model_path, data_path, output_dir="validation_results"):
    """Generate 10 sequences per year (2017-2024) and compute metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize components
    diffusion = DiffusionProcess(device=device)
    model = ConditionalUNet1D(seq_len=256).to(device)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    
    dataset = FinancialDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Define years
    years = list(range(2017, 2025))  # 2017 to 2024
    num_groups_per_year = 10
    
    # Get annual start dates
    annual_dates = dataset.get_annual_start_dates(years)  # [8, 3]
    
    # Storage for results
    metrics = {}
    all_gen_samples = []
    all_real_samples = []
    
    # Global real data metrics
    real_all = torch.cat([batch["sequence"] for batch in dataloader], dim=0)
    
    # Process each year
    for year_idx, year in enumerate(years):
        annual_date = annual_dates[year_idx].to(device)
        condition = {"date": annual_date.unsqueeze(0)}
        
        # Get real sequences for this year
        real_sequences, real_indices = dataset.get_real_sequences_for_year(
            annual_date.cpu(), num_samples=num_groups_per_year
        )
        
        # Generate 10 groups and compute metrics
        year_metrics_list = []
        year_gen_samples = []
        year_real_samples = []
        
        for group_idx in range(num_groups_per_year):
            # Generate one sequence
            gen_data = generate_samples(
                model, diffusion, condition,
                num_samples=1,
                device=device
            )
            
            # Get corresponding real data
            real_data = real_sequences[group_idx].unsqueeze(0)
            
            # Compute metrics for this group
            group_metrics = calculate_metrics(real_data, gen_data)
            year_metrics_list.append(group_metrics)
            
            # Store samples for visualization (first 3 groups)
            if group_idx < 3:
                year_real_samples.append(real_data.squeeze())
                year_gen_samples.append(gen_data.squeeze())
            
            # Collect for global metrics
            all_gen_samples.append(gen_data)
            all_real_samples.append(real_data)
        
        # Compute average metrics for the year
        year_metrics = {
            'real_mean': np.mean([m['real_mean'] for m in year_metrics_list]),
            'gen_mean': np.mean([m['gen_mean'] for m in year_metrics_list]),
            'real_std': np.mean([m['real_std'] for m in year_metrics_list]),
            'gen_std': np.mean([m['gen_std'] for m in year_metrics_list]),
            'real_corr': np.mean([m['real_corr'] for m in year_metrics_list]),
            'gen_corr': np.mean([m['gen_corr'] for m in year_metrics_list]),
            'real_acf': np.mean([m['real_acf'] for m in year_metrics_list]),
            'gen_acf': np.mean([m['gen_acf'] for m in year_metrics_list]),
            'kl_div': np.mean([m['kl_div'] for m in year_metrics_list])
        }
        metrics[f'year_{year}'] = year_metrics
        
        # Save visualizations and metrics for this year
        save_visualizations(year_real_samples, year_gen_samples, year_metrics, year, output_dir)
    
    # Compute global metrics
    gen_all = torch.cat(all_gen_samples, dim=0)
    real_all_subset = torch.cat(all_real_samples, dim=0)
    metrics['global'] = calculate_metrics(real_all_subset, gen_all)
    
    # Save global metrics
    with open(os.path.join(output_dir, 'metrics_global.json'), 'w') as f:
        json.dump(metrics['global'], f, indent=2)
    
    # Print report
    print_enhanced_report(metrics, years)
    print(f"\nValidation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    run_validation(
        model_path="saved_models/model_epoch_1000.pth",
        data_path="financial_data/sequences/sequences_256.pt"
    )