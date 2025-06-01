import json
import os
import numpy as np
import torch
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
    intermediate_samples = {100: [], 300: [], 500: [], 700: [], 900: []}
    # Adjust step_indices to match num_timesteps
    step_indices = torch.linspace(diffusion.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    for i in range(steps):
        t = step_indices[i]
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, labels)
        # DDPM update
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        alpha_t = diffusion.alphas[t].view(-1, 1)
        beta_t = diffusion.betas[t].view(-1, 1)
        x = (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        # Save intermediate samples at specific timesteps
        target_ts = [100, 300, 500, 700, 900]
        if int(t) in target_ts:
            intermediate_samples[int(t)].append(x.cpu())
            
    # Stack intermediate samples, handle empty cases
    gen_intermediate = {}
    for t in intermediate_samples:
        if intermediate_samples[t]:
            gen_intermediate[t] = torch.stack(intermediate_samples[t], dim=0)[:10]
        else:
            print(f"Warning: No samples saved for t={t}")
            gen_intermediate[t] = torch.zeros(10, num_samples, 256)  # Fallback
    
    return x.cpu(), gen_intermediate

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
    ks_stat, _ = stats.ks_2samp(real_data_np, gen_data_np)
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

def plot_spectrogram_comparison(real_sequences, gen_intermediate, output_path, num_samples=1000):
    """Plot power spectrum comparison between real and generated sequences."""
    plt.figure(figsize=(12, 8))
    
    # Compute power spectrum for real sequences
    real_power = []
    for seq in real_sequences[:num_samples]:
        fft_result = np.fft.rfft(seq.numpy() - seq.numpy().mean())  # Remove mean
        power = np.abs(fft_result) ** 2
        real_power.append(power)
    real_power = np.array(real_power)  # Shape: (num_samples, 129)
    
    # Normalize each sequence's power spectrum individually
    real_power = real_power / np.sum(real_power, axis=1, keepdims=True)  # Shape: (num_samples, 129)
    
    # Compute mean and quantiles after normalization
    mean_power = np.mean(real_power, axis=0)  # Shape: (129,)
    quantiles = np.quantile(real_power, [0.25, 0.75], axis=0)  # Shape: (2, 129)
    
    # Normalize frequencies
    freqs = np.linspace(0, 0.5, len(mean_power))
    
    # Plot real sequence power spectrum with shaded area and quantile lines
    plt.fill_between(freqs, 10 * np.log10(quantiles[0] + 1e-10), 10 * np.log10(quantiles[1] + 1e-10),
                     color='gray', alpha=0.1, label='Real 25%-75% Quantile')
    plt.plot(freqs, 10 * np.log10(quantiles[0] + 1e-10), color='gray', linestyle='--', linewidth=1.5, label='Real 25% Quantile')
    plt.plot(freqs, 10 * np.log10(quantiles[1] + 1e-10), color='gray', linestyle='--', linewidth=1.5, label='Real 75% Quantile')
    plt.plot(freqs, 10 * np.log10(mean_power + 1e-10), color='black', label='Real Mean', linewidth=2)

    # Compute power spectrum for generated intermediate sequences
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(gen_intermediate)))
    for i, (t, gen_seqs) in enumerate(sorted(gen_intermediate.items())):
        gen_power = []
        for seq in gen_seqs[:num_samples]:
            fft_result = np.fft.rfft(seq.numpy() - seq.numpy().mean())
            power = np.abs(fft_result) ** 2
            gen_power.append(power)
        gen_power = np.array(gen_power)  # Shape: (num_samples, 129)
        if gen_power.shape[0] > 0:  # Ensure non-empty
            gen_power = gen_power / np.sum(gen_power, axis=1, keepdims=True)  # Normalize
            mean_gen_power = np.mean(gen_power, axis=0)  # Shape: (129,)
            print(f"Timestep {t}: mean_gen_power shape = {mean_gen_power.shape}")
            plt.plot(freqs, 10 * np.log10(mean_gen_power.T + 1e-10), color=colors[i], 
                     label=f'Gen t={t}', linewidth=1.5, alpha=0.7)
    
    # Customize plot
    plt.xlabel('Frequency (cycles/day)')
    plt.ylabel('Power (dB)')
    plt.ylim(-60, 0)
    plt.xlim(0, 0.5)
    plt.title('Power Spectrum Comparison: Real vs Generated Sequences')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

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
    
    # Sample real sequences for spectrogram baseline
    real_sequences = []
    for year in years:
        year_idx = year - 2017
        annual_date = annual_dates[year_idx].to(device)
        real_seqs, _ = dataset.get_real_sequences_for_year(annual_date.cpu(), num_samples=10)
        real_sequences.extend(real_seqs)
    real_sequences = torch.stack(real_sequences)
    
    for year in years:
        year_idx = year - 2017  # Compute index for annual_dates
        annual_date = annual_dates[year_idx].to(device)
        condition = {"date": annual_date.unsqueeze(0)}
        real_sequences_year, _ = dataset.get_real_sequences_for_year(
            annual_date.cpu(), num_samples=num_groups_per_year)
        year_metrics_list = []
        year_gen_samples = []
        year_real_samples = []  # Initialize list
        gen_data, gen_intermediate = generate_samples(
            model, diffusion, condition,
            num_samples=num_groups_per_year,
            device=device,
            steps=1000  # Match num_timesteps
        )
        # Inverse scale generated and real data
        gen_data = dataset.inverse_scale(gen_data)
        real_sequences_year = dataset.inverse_scale(real_sequences_year)
        for group_idx in range(num_groups_per_year):
            group_data = gen_data[group_idx:group_idx+1]
            real_data = real_sequences_year[group_idx].unsqueeze(0)
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
    
    # Generate spectrogram comparison
    output_dir = config["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    plot_spectrogram_comparison(real_sequences, gen_intermediate, 
                               os.path.join(output_dir, 'spectrogram_comparison.png'))
    print(f"\nSpectrogram comparison saved to {os.path.join(output_dir, 'spectrogram_comparison.png')}")
    print(f"\nValidation complete! Results saved to {config['save_dir']}")

if __name__ == "__main__":
    config = {
        "model_path": "saved_models/model_epoch_500.pth",
        "data_path": "financial_data/sequences/sequences_256.pt",
        "save_dir": "validation_results",
        "channels": [32, 64, 128, 512, 1024, 2048]
    }
    os.makedirs(config["save_dir"], exist_ok=True)
    run_validation(config)