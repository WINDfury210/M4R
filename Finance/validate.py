"""
Diffusion Model Validation Script
Generate 10 sequences per year (2017-2024) with DDPM and compute per-sample metrics
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf

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
        self.ln1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = out.permute(0, 2, 1)  # [batch_size, seq_len, out_channels]
        out = self.ln1(out)
        out = out.permute(0, 2, 1)  # [batch_size, out_channels, seq_len]
        out = self.relu(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = out.permute(0, 2, 1)
        out += residual
        return self.relu(out)

class ConditionalUNet1D(nn.Module):
    def __init__(self, seq_len=256, channels=[32, 64, 128, 256]):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_levels = len(channels)
        
        self.time_embed = TimeEmbedding(dim=channels[-1], embedding_type="sinusoidal")
        self.date_embed = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, channels[-1]),
            nn.LayerNorm(channels[-1])
        )
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        self.encoder_convs = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_channels = channels[0]
        for i, out_channels in enumerate(channels):
            self.encoder_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                              stride=2 if i>0 else 1, padding=1))
            self.encoder_res.append(ResidualBlock1D(out_channels, out_channels))
            self.attentions.append(SelfAttention1D(out_channels) if 0<i<len(channels) else nn.Identity())
            in_channels = out_channels
        self.mid_conv1 = ResidualBlock1D(channels[-1], channels[-1])
        self.mid_conv2 = ResidualBlock1D(channels[-1], channels[-1])
        self.decoder_convs = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        for i in range(len(channels)-1):
            in_channels = channels[-1-i] + channels[-1-i]
            out_channels = channels[-2-i]
            self.decoder_convs.append(nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ))
            self.decoder_res.append(ResidualBlock1D(out_channels, out_channels))
        self.final_conv = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x, t, date):
        time_emb = self.time_embed(t)
        date_emb = self.date_embed(date)
        combined_cond = time_emb + date_emb # Disable date embedding
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        skips = []
        for i, (conv, res, attn) in enumerate(zip(self.encoder_convs, self.encoder_res, self.attentions)):
            x = F.relu(conv(x))
            x = res(x)
            x = attn(x)
            skips.append(x)
        x = self.mid_conv1(x)
        cond = combined_cond.unsqueeze(-1)
        x = x + cond
        x = self.mid_conv2(x)
        for i, (conv, res) in enumerate(zip(self.decoder_convs, self.decoder_res)):
            skip = skips[-(i+1)]
            if x.shape[-1] != skip.shape[-1]:
                if x.shape[-1] < skip.shape[-1]:
                    pad_len = skip.shape[-1] - x.shape[-1]
                    x = F.pad(x, (0, pad_len))
                else:
                    x = x[:, :, :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = F.relu(conv(x))
            x = res(x)
        x = self.final_conv(x).squeeze(1)
        return x
    
# 2. Diffusion Process -------------------------------------------------------

class DiffusionProcess:
    def __init__(self, num_timesteps=200, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self._linear_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def _linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.num_timesteps)
    
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
        min_year, max_year = 2017, 2024
        start_dates = []
        for year in years:
            norm_year = (year - min_year) / (max_year - min_year)
            start_date = torch.tensor([norm_year, 0.0, 0.0], dtype=torch.float32)
            start_dates.append(start_date)
        return torch.stack(start_dates)
    
    def get_real_sequences_for_year(self, annual_date, num_samples=10):
        date_diffs = torch.norm(self.dates - annual_date, dim=1)
        closest_indices = torch.argsort(date_diffs)[:num_samples]
        return self.sequences[closest_indices], closest_indices

# 4. Validation Core ---------------------------------------------------------

@torch.no_grad()
def generate_samples(model, diffusion, conditions, num_samples=1, device="cuda", steps=1500):
    """Generate samples with DDPM sampling"""
    model.eval()
    labels = conditions["date"].repeat(num_samples, 1).to(device)
    year = conditions["date"][0, 0].item() * (2024 - 2017) + 2017
    # noise_scale = 0.05 if year in [2019, 2020, 2021] else 0.01
    x = torch.randn(num_samples, 256, device=device) # * noise_scale
    for t in reversed(range(steps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, labels)
        
        alpha_t = diffusion.alphas[t].view(-1, 1)
        beta_t = diffusion.betas[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        
        x = (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        
        x = torch.clamp(x, -3, 3)
    
    print(f"Year {int(year)}: Gen mean={x.mean().item():.6f}, std={x.std().item():.6f}")
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
    
    real_data_np = real_data.numpy().flatten()
    gen_data_np = generated_data.numpy().flatten()
    
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
    metrics['ks_pval'] = ks_pval
    
    # Wasserstein Distance
    wass_dist = stats.wasserstein_distance(real_data_np, gen_data_np)
    metrics['wass_dist'] = wass_dist
    
    # Shapiro-Wilk Test
    _, shapiro_pval = stats.shapiro(gen_data_np)
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
    print(f"{'Metric':<12} {'Diff':>10} {'Z-score':>10}")
    print("-" * 34)
    global_stats = metrics_dict['global']
    z_score_mean = (global_stats['gen_mean'] - global_stats['real_mean']) / global_stats['real_std']
    z_score_std = (global_stats['gen_std'] - global_stats['real_std']) / global_stats['real_std']
    print(f"{'Mean':<12} {abs(global_stats['gen_mean']-global_stats['real_mean']):>10.6f} {z_score_mean:>10.2f}")
    print(f"{'Std Dev':<12} {abs(global_stats['gen_std']-global_stats['real_std']):>10.6f} {z_score_std:>10.2f}")
    print(f"{'Corr':<12} {abs(global_stats['gen_corr']-global_stats['real_corr']):>10.6f} {'-':>10}")
    print(f"{'Autocorr':<12} {abs(global_stats['gen_acf']-global_stats['real_acf']):>10.6f} {'-':>10}")
    print(f"{'KS Stat':<12} {global_stats['ks_stat']:>10.6f} {'-':>10}")
    print(f"{'Wass Dist':<12} {global_stats['wass_dist']:>10.6f} {'-':>10}")
    print(f"{'Shapiro P':<12} {global_stats['shapiro_pval']:>10.6f} {'-':>10}")
    print(f"{'Skew Diff':<12} {abs(global_stats['gen_skew']-global_stats['real_skew']):>10.6f} {'-':>10}")
    print(f"{'Kurt Diff':<12} {abs(global_stats['gen_kurt']-global_stats['real_kurt']):>10.6f} {'-':>10}")
    
    # Per-Sample Statistics (First 3 samples per year)
    print("\n[Per-Sample Statistics]")
    print(f"{'Year':<6} {'Sample':<8} {'Mean Diff':>10} {'Std Diff':>10} {'Corr Diff':>10} {'Acf Diff':>10} {'KS Stat':>10} {'Wass Dist':>10}")
    print("-" * 74)
    for year in years:
        for i, sample_metrics in enumerate(metrics_dict[f'year_{year}'][:3]):
            diff_values = []
            for metric in ['mean', 'std', 'corr', 'acf', 'ks_stat', 'wass_dist']:
                if metric in ['ks_stat', 'wass_dist']:
                    diff = sample_metrics[metric]
                else:
                    real_key = f'real_{metric}'
                    gen_key = f'gen_{metric}'
                    diff = abs(sample_metrics[gen_key] - sample_metrics[real_key])
                # 高亮阈值：Mean 和 Acf > 0.0002，Std 和 Corr > 0.005，KS > 0.05，Wass > 0.01
                threshold = 0.0002 if metric == 'mean' else 0.005 if metric in ['std', 'corr', 'acf'] else 0.05 if metric == 'ks_stat' else 0.01
                diff_values.append(f"{diff:>8.6f}{'*' if diff > threshold else ''}")
            print(f"{year:<6} {i+1:<8} {diff_values[0]:>10} {diff_values[1]:>10} {diff_values[2]:>10} {diff_values[3]:>10} {diff_values[4]:>10} {diff_values[5]:>10}")

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

def run_validation(model_path, data_path, output_dir="validation_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = DiffusionProcess(device=device, num_timesteps=1500)
    model = ConditionalUNet1D(seq_len=256, channels=[64, 128, 256, 512, 1024]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    dataset = FinancialDataset(data_path)
    years = list(range(2017, 2025))
    num_groups_per_year = 10
    annual_dates = dataset.get_annual_start_dates(years)
    metrics = {}
    all_gen_samples = []
    all_real_samples = []
    
    for year_idx, year in enumerate(years):
        annual_date = annual_dates[year_idx].to(device)
        condition = {"date": annual_date.unsqueeze(0)}
        real_sequences, _ = dataset.get_real_sequences_for_year(
            annual_date.cpu(), num_samples=num_groups_per_year
        )
        year_metrics_list = []
        year_gen_samples = []
        year_real_samples = []
        gen_data = generate_samples(
            model, diffusion, condition,
            num_samples=num_groups_per_year,
            device=device,
            steps=200
        )
        for group_idx in range(num_groups_per_year):
            group_data = gen_data[group_idx:group_idx+1]
            real_data = real_sequences[group_idx].unsqueeze(0)
            group_metrics = calculate_metrics(real_data, group_data)
            year_metrics_list.append(group_metrics)
            if group_idx < 3:
                year_real_samples.append(real_data.squeeze())
                year_gen_samples.append(group_data.squeeze())
            all_gen_samples.append(group_data)
            all_real_samples.append(real_data)
        metrics[f'year_{year}'] = year_metrics_list
        # save_visualizations(year_metrics_list, year_gen_samples, year_metrics_list, year, output_dir)
    
    gen_all = torch.cat(all_gen_samples, dim=0)
    real_all_subset = torch.cat(all_real_samples, dim=0)
    metrics['global'] = calculate_metrics(real_all_subset, gen_all)
    print_enhanced_report(metrics, years)
    print(f"\nValidation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    run_validation(
        model_path="saved_models/final_model.pth",
        data_path="financial_data/sequences/sequences_256.pt"
    )