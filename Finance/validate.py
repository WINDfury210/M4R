import json
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from scipy import stats
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from model import *

# 1. Diffusion Process -------------------------------------------------------

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu"):
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

# 2. Data Loading ------------------------------------------------------------

class FinancialDataset(Dataset):
    def __init__(self, data_path, scale_factor=1.0):
        data = torch.load(data_path)
        self.sequences = data["sequences"]
        self.dates = data["start_dates"]
        self.original_mean = self.sequences.mean().item()
        self.original_std = self.sequences.std().item()
        self.scale_factor = scale_factor
        self.sequences = (self.sequences - self.original_mean) / self.original_std * scale_factor
        print(f"Scaled data - Mean: {self.sequences.mean().item():.6f}, Std: {self.sequences.std().item():.6f}")
        print(f"Total sequences: {len(self.sequences)}")
        print(f"Sequences shape: {self.sequences.shape}")
        print(f"Dates shape: {self.dates.shape}")
        print(f"Dates[:, 0] range: {self.dates[:, 0].min().item():.6f} to {self.dates[:, 0].max().item():.6f}")
        unique_years = torch.unique(self.dates[:, 0]).tolist()
        print(f"Unique dates[:, 0] values: {unique_years[:10]}{'...' if len(unique_years) > 10 else ''}")
    
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
            norm_year = (year - min_year) / 8.0
            start_date = torch.tensor([norm_year, 0.0, 0.0], dtype=torch.float32)
            start_dates.append(start_date)
        return torch.stack(start_dates)
    
    def get_random_dates_for_year(self, year, num_samples):
        min_year, max_year = 2017, 2024
        norm_year = (year - min_year) / 8.0
        random_dates = []
        for _ in range(num_samples):
            norm_month = random.uniform(0, 1)
            norm_day = random.uniform(0, 1)
            random_date = torch.tensor([norm_year, norm_month, norm_day], dtype=torch.float32)
            random_dates.append(random_date)
        return torch.stack(random_dates)
    
    def inverse_scale(self, sequences):
        return sequences * self.original_std / self.scale_factor + self.original_mean

# 3. Validation Core ---------------------------------------------------------

@torch.no_grad()
def generate_samples(model, diffusion, condition, num_samples, device, steps=1000, step_interval=50):
    model.eval()
    labels = condition["date"].to(device)
    x = torch.randn(num_samples, 256, device=device)
    intermediate_samples = {}
    step_indices = torch.linspace(diffusion.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    
    # Determine target timesteps for saving (every step_interval)
    target_ts = list(range(0, diffusion.num_timesteps + 1, step_interval))[::-1]
    
    for i in range(steps):
        t = step_indices[i]
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, labels)
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        alpha_t = diffusion.alphas[t].view(-1, 1)
        beta_t = diffusion.betas[t].view(-1, 1)
        x = (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        
        # Save intermediate samples at target timesteps
        if int(t+1) in target_ts:
            intermediate_samples[int(t+1)] = x.cpu()
    
    gen_intermediate = {}
    for t in intermediate_samples:
        gen_intermediate[t] = intermediate_samples[t].unsqueeze(0)
    
    return x.cpu(), gen_intermediate

def calculate_metrics(data, dummy=None):
    metrics = {}
    
    if data.dim() == 1:
        data = data.unsqueeze(0)
    
    # Check if data is valid
    if data.numel() == 0 or torch.all(data == 0):
        print(f"Warning: Invalid data in calculate_metrics (shape: {data.shape}, all zeros: {torch.all(data == 0)})")
        return {
            'gen_mean': 0.0, 'gen_std': 0.0, 'gen_corr': 0.0, 'gen_acf': 0.0,
            'gen_skew': 0.0, 'gen_kurt': 0.0,
            'abs_gen_mean': 0.0, 'abs_gen_std': 0.0, 'abs_gen_corr': 0.0,
            'abs_gen_acf': 0.0, 'abs_gen_skew': 0.0, 'abs_gen_kurt': 0.0
        }
    
    metrics['gen_mean'] = data.mean().item()
    metrics['gen_std'] = data.std().item() if data.numel() > 1 else 0.0
    
    data_np = data.cpu().numpy().flatten()
    lagged = data_np[:-1]
    next_val = data_np[1:]
    metrics['gen_corr'] = np.corrcoef(lagged, next_val)[0, 1] if len(lagged) > 1 else 0.0
    
    try:
        gen_acf = acf(data_np, nlags=20, fft=True)[1:].mean()
        if np.isnan(gen_acf):
            gen_acf = 0.0
    except Exception as e:
        print(f"Warning: ACF computation failed, setting gen_acf to 0.0: {e}")
        gen_acf = 0.0
    metrics['gen_acf'] = gen_acf
    
    metrics['gen_skew'] = stats.skew(data_np) if len(data_np) > 2 else 0.0
    metrics['gen_kurt'] = stats.kurtosis(data_np) if len(data_np) > 3 else 0.0
    
    abs_data = torch.abs(data)
    abs_data_np = abs_data.cpu().numpy().flatten()
    
    metrics['abs_gen_mean'] = abs_data.mean().item()
    metrics['abs_gen_std'] = abs_data.std().item() if abs_data.numel() > 1 else 0.0
    
    abs_lagged = abs_data_np[:-1]
    abs_next = abs_data_np[1:]
    metrics['abs_gen_corr'] = np.corrcoef(abs_lagged, abs_next)[0, 1] if len(abs_lagged) > 1 else 0.0
    
    try:
        abs_gen_acf = acf(abs_data_np, nlags=20, fft=True)[1:].mean()
        if np.isnan(abs_gen_acf):
            abs_gen_acf = 0.0
    except Exception as e:
        print(f"Warning: Abs ACF computation failed, setting abs_gen_acf to 0.0: {e}")
        abs_gen_acf = 0.0
    metrics['abs_gen_acf'] = abs_gen_acf
    
    metrics['abs_gen_skew'] = stats.skew(abs_data_np) if len(abs_data_np) > 2 else 0.0
    metrics['abs_gen_kurt'] = stats.kurtosis(abs_data_np) if len(abs_data_np) > 3 else 0.0
    
    return metrics

def average_metrics(metrics_list):
    if not metrics_list:
        return {
            'gen_mean': {'mean': 0.0, 'variance': 0.0},
            'gen_std': {'mean': 0.0, 'variance': 0.0},
            'gen_corr': {'mean': 0.0, 'variance': 0.0},
            'gen_acf': {'mean': 0.0, 'variance': 0.0},
            'gen_skew': {'mean': 0.0, 'variance': 0.0},
            'gen_kurt': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_mean': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_std': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_corr': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_acf': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_skew': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_kurt': {'mean': 0.0, 'variance': 0.0}
        }
    
    stats = {}
    keys = list(metrics_list[0].keys())
    for key in keys:
        values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        if values:
            stats[key] = {
                'mean': float(np.mean(values)),
                'variance': float(np.var(values))
            }
        else:
            stats[key] = {'mean': 0.0, 'variance': 0.0}
    return stats

def save_visualizations(gen_samples, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if not gen_samples:
        print(f"Warning: No generated samples for year {year}, skipping visualization")
        return
    
    # 随机选择一个样本
    idx = random.randint(0, len(gen_samples) - 1)
    gen_sample = gen_samples[idx].numpy()
    abs_gen_sample = np.abs(gen_sample)
    
    gen_mean = np.mean(gen_sample)
    gen_std = np.std(gen_sample) if len(gen_sample) > 1 else 0.0
    y_min_gen = gen_mean - 3 * gen_std
    y_max_gen = gen_mean + 3 * gen_std
    
    abs_gen_mean = np.mean(abs_gen_sample)
    abs_gen_std = np.std(abs_gen_sample) if len(abs_gen_sample) > 1 else 0.0
    y_min_abs = max(0, abs_gen_mean - 3 * abs_gen_std)
    y_max_abs = abs_gen_mean + 3 * abs_gen_std
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(gen_sample, label="Generated", color='blue')
    plt.title(f"Generated Sample (Year {year})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.ylim(y_min_gen, y_max_gen)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    try:
        gen_acf = acf(gen_sample, nlags=20, fft=True)
    except:
        gen_acf = np.zeros(21)
    plt.stem(range(len(gen_acf)), gen_acf, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title(f"Autocovariance (Year {year})")
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.ylim(min(gen_acf) - 0.1, max(gen_acf) + 0.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_original_sample.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(abs_gen_sample, label="Abs Generated", color='green')
    plt.title(f"Abs Generated Sample (Year {year})")
    plt.xlabel("Time")
    plt.ylabel("Absolute Value")
    plt.ylim(y_min_abs, y_max_abs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    try:
        abs_gen_acf = acf(abs_gen_sample, nlags=20, fft=True)
    except:
        abs_gen_acf = np.zeros(21)
    plt.stem(range(len(abs_gen_acf)), abs_gen_acf, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title(f"Abs Autocovariance (Year {year})")
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.ylim(min(abs_gen_acf) - 0.1, max(abs_gen_acf) + 0.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_absolute_sample.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    stats.probplot(gen_sample, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot (Year {year})")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.ylim(y_min_gen, y_max_gen)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_qq_plot.png'), dpi=300)
    plt.close()

def plot_metrics_vs_timesteps(metrics_per_timestep, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    timesteps = sorted(metrics_per_timestep.keys(), reverse=True)  # From 1000 to 0
    if not timesteps:
        print("Warning: No metrics available for timesteps")
        return
    
    metrics_to_plot = ['gen_mean', 'gen_std', 'gen_kurt', 'abs_gen_mean', 'abs_gen_std', 'abs_gen_kurt']
    
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics_to_plot, 1):
        means = [metrics_per_timestep[t].get(metric, {}).get('mean', 0.0) for t in timesteps]
        variances = [metrics_per_timestep[t].get(metric, {}).get('variance', 0.0) for t in timesteps]
        
        plt.subplot(2, 3, i)
        plt.plot(timesteps, means, marker='o', color='blue', label='Mean')
        plt.fill_between(timesteps,
                         [m - np.sqrt(v) for m, v in zip(means, variances)],
                         [m + np.sqrt(v) for m, v in zip(means, variances)],
                         color='blue', alpha=0.2, label='±1 Std Dev')
        plt.title(metric)
        plt.xlabel('Timestep')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_timesteps.png'), dpi=300)
    plt.close()

def print_enhanced_report(metrics_dict, years):
    print("\n=== Validation Report ===")
    
    print("\n[Global Statistics]")
    print(f"{'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 39)
    global_stats = metrics_dict['global']
    for metric in ['gen_mean', 'gen_std', 'gen_corr', 'gen_acf', 'gen_skew', 'gen_kurt']:
        mean = global_stats.get(metric, {}).get('mean', 0.0)
        variance = global_stats.get(metric, {}).get('variance', 0.0)
        print(f"{metric:<15} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Absolute Value Global Statistics]")
    print(f"{'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 39)
    for metric in ['abs_gen_mean', 'abs_gen_std', 'abs_gen_corr', 'abs_gen_acf', 'abs_gen_skew', 'abs_gen_kurt']:
        mean = global_stats.get(metric, {}).get('mean', 0.0)
        variance = global_stats.get(metric, {}).get('variance', 0.0)
        print(f"{metric:<15} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Yearly Statistics]")
    print(f"{'Year':<6} {'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 45)
    for year in years:
        year_stats = metrics_dict.get(f'year_{year}', {})
        for metric in ['gen_mean', 'gen_std', 'gen_corr', 'gen_acf', 'gen_skew', 'gen_kurt']:
            mean = year_stats.get(metric, {}).get('mean', 0.0)
            variance = year_stats.get(metric, {}).get('variance', 0.0)
            print(f"{year:<6} {metric:<15} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Absolute Value Yearly Statistics]")
    print(f"{'Year':<6} {'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 45)
    for year in years:
        year_stats = metrics_dict.get(f'year_{year}', {})
        for metric in ['abs_gen_mean', 'abs_gen_std', 'abs_gen_corr', 'abs_gen_acf', 'abs_gen_skew', 'abs_gen_kurt']:
            mean = year_stats.get(metric, {}).get('mean', 0.0)
            variance = year_stats.get(metric, {}).get('variance', 0.0)
            print(f"{year:<6} {metric:<15} {mean:>12.6f} {variance:>12.6f}")

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
    dataset = FinancialDataset(config["data_path"], scale_factor=1.0)
    years = list(range(2017, 2024))  # 2017-2023
    num_groups_per_year = config.get("num_groups_per_year", 10)
    step_interval = config.get("step_interval", 50)
    metrics = {}
    all_gen_samples = []
    metrics_per_timestep = {}
    
    for year in years:
        random_dates = dataset.get_random_dates_for_year(year, num_groups_per_year).to(device)
        year_metrics_list = []
        year_gen_samples = []
        
        for i in range(num_groups_per_year):
            condition = {"date": random_dates[i].unsqueeze(0)}
            gen_data, gen_intermediate = generate_samples(
                model, diffusion, condition,
                num_samples=1,
                device=device,
                steps=1000,
                step_interval=step_interval
            )
            print(f"Year {year}, Sample {i}: gen_data shape {gen_data.shape}, min {gen_data.min().item():.6f}, max {gen_data.max().item():.6f}")
            gen_data = dataset.inverse_scale(gen_data)
            gen_metrics = calculate_metrics(gen_data)
            year_metrics_list.append(gen_metrics)
            year_gen_samples.append(gen_data.squeeze())
            all_gen_samples.append(gen_data.squeeze())
            
            # Compute metrics for intermediate samples
            for t in gen_intermediate:
                inter_samples = dataset.inverse_scale(gen_intermediate[t].squeeze(0))
                inter_metrics = calculate_metrics(inter_samples)
                if t not in metrics_per_timestep:
                    metrics_per_timestep[t] = []
                metrics_per_timestep[t].append(inter_metrics)
        
        # Compute average metrics
        metrics[f'year_{year}'] = average_metrics(year_metrics_list)
        
        # Save visualizations for one random sample
        save_visualizations(year_gen_samples, year, config["save_dir"])
    
    # Compute global metrics
    gen_all = torch.cat(all_gen_samples, dim=0)
    metrics['global'] = average_metrics([calculate_metrics(gen_all[i].unsqueeze(0)) for i in range(gen_all.shape[0])])
    
    # Average metrics per timestep
    for t in metrics_per_timestep:
        metrics_per_timestep[t] = average_metrics(metrics_per_timestep[t])
    
    # Print report
    print_enhanced_report(metrics, years)
    
    # Plot metrics vs timesteps
    plot_metrics_vs_timesteps(metrics_per_timestep, config["save_dir"])
    
    output_dir = config["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nMetrics vs timesteps plot saved to {os.path.join(output_dir, 'metrics_vs_timesteps.png')}")
    print(f"\nValidation complete! Results saved to {config['save_dir']}")

if __name__ == "__main__":
    config = {
        "model_path": "saved_models/final_model.pth",
        "data_path": "financial_data/sequences/sequences_256.pt",
        "save_dir": "validation_results",
        "channels": [32, 128, 512, 2048],
        "num_groups_per_year": 100,
        "step_interval": 10
    }
    os.makedirs(config["save_dir"], exist_ok=True)
    run_validation(config)