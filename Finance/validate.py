"""
Corrected Diffusion Model Validation Script
Adapted for ConditionalUNet1D with seq_len=256 and new condition embedding
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
            nn.Linear(512, channels[-1])
        )
        
        # Market cap embedding (1D -> channels[-1] = 256)
        self.mcap_embed = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, channels[-1])
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

    def forward(self, x, t, date, market_cap):
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
        time_emb = self.time_embed(t)  # [batch_size, 256]
        date_emb = self.date_embed(date_emb)  # [batch_size, 256]
        mcap_emb = self.mcap_embed(market_cap)  # [batch_size, 256]
        combined_cond = time_emb + date_emb + mcap_emb  # [batch_size, 256]
        
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
        
        # Cosine beta schedule
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
        self.market_caps = data["market_caps"]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx],
            "market_cap": self.market_caps[idx]
        }

# 4. Validation Core ---------------------------------------------------------

def generate_samples(model, diffusion, conditions, num_samples=100, steps=200, device="cuda"):
    """Improved sample generation function"""
    with torch.no_grad():
        # Prepare condition data
        dates = conditions["date"].repeat(num_samples, 1).to(device)
        market_caps = conditions["market_cap"].repeat(num_samples, 1).to(device)
        
        # Initialize noise (reduced scale)
        samples = torch.randn(num_samples, 256, device=device) * 0.1
        
        # Improved generation process
        for t in tqdm(reversed(range(0, diffusion.num_timesteps, diffusion.num_timesteps // steps)),
                     desc="Generating samples"):
            times = torch.full((num_samples,), t, device=device, dtype=torch.long)
            pred_noise = model(samples, times, dates, market_caps)
            
            # Noise scaling
            alpha_bar = diffusion.alpha_bars[t]
            alpha_bar_prev = diffusion.alpha_bars[t-1] if t > 0 else torch.tensor(1.0)
            
            # Stable update rule
            x0_pred = (samples - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            eps_coef = (1 - alpha_bar_prev) / (1 - alpha_bar)
            eps_coef = torch.clamp(eps_coef, max=2.0)
            
            samples = x0_pred * alpha_bar_prev.sqrt() + \
                     eps_coef.sqrt() * pred_noise * (1 - alpha_bar_prev).sqrt()
        
        # Post-processing: clip outliers
        samples = torch.clamp(samples, min=-3, max=3)
        return samples.cpu()

def print_enhanced_report(metrics_dict, num_conditions):
    """Enhanced statistical report"""
    print("\n=== Enhanced Validation Report ===")
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
    
    print("\n[Condition Statistics (First 3)]")
    for i in range(min(3, num_conditions)):
        cond_stats = metrics_dict[f'condition_{i}']
        z_mean = (cond_stats['gen_mean'] - cond_stats['real_mean']) / cond_stats['real_std']
        z_std = (cond_stats['gen_std'] - cond_stats['real_std']) / cond_stats['real_std']
        
        print(f"\nCondition {i+1}:")
        print(f"{'Mean':<15} | {cond_stats['real_mean']:>12.6f} | {cond_stats['gen_mean']:>12.6f} | "
              f"{abs(cond_stats['gen_mean']-cond_stats['real_mean'])/cond_stats['real_mean']*100:>11.2f}% | {z_mean:>10.2f}")
        print(f"{'Std Dev':<15} | {cond_stats['real_std']:>12.6f} | {cond_stats['gen_std']:>12.6f} | "
              f"{abs(cond_stats['gen_std']-cond_stats['real_std'])/cond_stats['real_std']*100:>11.2f}% | {z_std:>10.2f}")
    
    print("\n[Issue Diagnostics]")
    if global_stats['gen_std'] > global_stats['real_std'] * 1.5:
        print("⚠️ Excessive generated volatility: Check noise schedule or reduce initial noise scale")
    if abs(z_score_mean) > 3:
        print("⚠️ Significant mean shift: Check condition injection or model bias")
    if global_stats['gen_corr'] < global_stats['real_corr'] * 0.5:
        print("⚠️ Low correlation: Consider increasing model capacity or adjusting training objective")

def calculate_metrics(real_data, generated_data):
    """Calculate comparison metrics"""
    metrics = {}
    
    # Basic statistics
    metrics['real_mean'] = real_data.mean().item()
    metrics['gen_mean'] = generated_data.mean().item()
    metrics['real_std'] = real_data.std().item()
    metrics['gen_std'] = generated_data.std().item()
    
    # Correlation
    real_data_np = real_data.numpy()
    gen_data_np = generated_data.numpy()
    real_corr = np.corrcoef(real_data_np, rowvar=False)
    gen_corr = np.corrcoef(gen_data_np, rowvar=False)
    metrics['real_corr'] = np.abs(real_corr).mean() if real_corr.shape[0] > 1 else 0.0
    metrics['gen_corr'] = np.abs(gen_corr).mean() if gen_corr.shape[0] > 1 else 0.0
    
    # Distribution similarity
    real_hist = np.histogram(real_data_np.flatten(), bins=50, density=True)[0]
    gen_hist = np.histogram(gen_data_np.flatten(), bins=50, density=True)[0]
    metrics['kl_div'] = F.kl_div(
        torch.from_numpy(gen_hist + 1e-10).log(),
        torch.from_numpy(real_hist + 1e-10),
        reduction='batchmean'
    ).item()
    
    return metrics

def print_comparison_report(metrics_dict, num_conditions):
    """Print detailed comparison report"""
    print("\n=== Validation Report ===")
    
    print("\n[Global Statistics]")
    print(f"{'Metric':<15} | {'Real':>10} | {'Generated':>10} | {'Difference':>10}")
    print("-" * 50)
    for metric in ['mean', 'std', 'corr']:
        real = metrics_dict['global'][f'real_{metric}']
        gen = metrics_dict['global'][f'gen_{metric}']
        diff = abs(real - gen)
        print(f"{metric:<15} | {real:>10.4f} | {gen:>10.4f} | {diff:>10.4f}")
    print(f"KL Divergence: {metrics_dict['global']['kl_div']:.4f}")
    
    print("\n[Condition-Specific Statistics]")
    for i in range(num_conditions):
        cond_metrics = metrics_dict[f'condition_{i}']
        print(f"\nCondition {i+1}:")
        print(f"{'Metric':<15} | {'Real':>10} | {'Generated':>10} | {'Diff %':>10}")
        print("-" * 50)
        for metric in ['mean', 'std']:
            real = cond_metrics[f'real_{metric}']
            gen = cond_metrics[f'gen_{metric}']
            diff_pct = abs(real - gen) / real * 100 if real != 0 else 0.0
            print(f"{metric:<15} | {real:>10.4f} | {gen:>10.4f} | {diff_pct:>9.2f}%")
        print(f"{'correlation':<15} | {cond_metrics['real_corr']:>10.4f} | "
              f"{cond_metrics['gen_corr']:>10.4f} | {'-':>10}")

def save_visualizations(real_samples, gen_samples, metrics, output_dir):
    """Save comparison visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Sample comparison plot
    plt.figure(figsize=(15, 5))
    for i in range(min(3, len(real_samples))):
        plt.subplot(2, 3, i+1)
        plt.plot(real_samples[i].numpy())
        plt.title(f"Real Sample {i+1}")
        
        plt.subplot(2, 3, i+4)
        plt.plot(gen_samples[i].numpy())
        plt.title(f"Generated Sample {i+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_comparison.png'))
    plt.close()

# 5. Main Validation Function ------------------------------------------------

def run_validation(model_path, data_path, output_dir="validation_results"):
    """Run complete validation pipeline"""
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
    
    # Prepare test conditions
    test_data = next(iter(dataloader))
    num_conditions = 5
    cond_indices = torch.randperm(len(test_data["date"]))[:num_conditions]
    
    # Storage for results
    metrics = {}
    real_samples = []
    gen_samples = []
    
    # Global real data metrics
    real_all = torch.cat([batch["sequence"] for batch in dataloader], dim=0)
    metrics['global'] = calculate_metrics(real_all, real_all)  # Baseline
    
    # Validate per condition
    for i, idx in enumerate(cond_indices):
        condition = {
            "date": test_data["date"][idx].unsqueeze(0),
            "market_cap": test_data["market_cap"][idx].unsqueeze(0)
        }
        
        # Get real data
        real_data = test_data["sequence"][idx].unsqueeze(0)
        
        # Generate samples
        gen_data = generate_samples(
            model, diffusion, condition,
            num_samples=100,
            device=device
        )
        
        # Calculate and store metrics
        metrics[f'condition_{i}'] = calculate_metrics(real_data, gen_data)
        
        # Store samples for visualization
        if i < 3:
            real_samples.append(real_data.squeeze())
            gen_samples.append(gen_data[0])
    
    # Calculate global generated metrics
    gen_all = torch.cat([
        generate_samples(
            model, diffusion,
            {"date": test_data["date"][i].unsqueeze(0),
             "market_cap": test_data["market_cap"][i].unsqueeze(0)},
            num_samples=20,
            device=device
        ) for i in cond_indices
    ])
    metrics['global'] = calculate_metrics(real_all, gen_all)
    
    # Print and save results
    print_enhanced_report(metrics, num_conditions)
    print_comparison_report(metrics, num_conditions)
    save_visualizations(real_samples, gen_samples, metrics, output_dir)
    print(f"\nValidation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    run_validation(
        model_path="saved_models/final_model.pth",
        data_path="financial_data/sequences/sequences_256.pt"
    )