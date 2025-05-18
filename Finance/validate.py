"""
Complete Diffusion Model Validation Pipeline

Features:
1. Contains all necessary components (model, diffusion, dataset)
2. Detailed metrics comparison between real and generated samples
3. Condition-specific analysis
4. Clean Pythonic style without type hints
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


# 1. Model Definitions --------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
    def forward(self, time):
        emb = time.float().view(-1, 1) * self.emb.view(1, -1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConditionalUNet(nn.Module):
    def __init__(self, seq_len=252):
        super().__init__()
        self.seq_len = seq_len
        
        # Time and condition embedding
        self.time_embed = TimeEmbedding(128)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Network architecture
        self.input_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        self.mid_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        
        self.output_conv = nn.Conv1d(32, 1, kernel_size=1)
        
        # Condition injection
        self.cond_proj_input = nn.Linear(128, 32)
        self.cond_proj_down1 = nn.Linear(128, 64)
        self.cond_proj_down2 = nn.Linear(128, 128)

    def forward(self, x, t, date, market_cap):
        # Embed conditions
        t_emb = self.time_embed(t)
        cond = torch.cat([date, market_cap], dim=-1)
        c_emb = self.cond_proj(cond)
        combined_cond = t_emb + c_emb
        
        # Process input
        x = x.unsqueeze(1)
        h0 = self.input_conv(x) + self.cond_proj_input(combined_cond).unsqueeze(-1)
        
        # Downsample
        h1 = self.down1(h0) + self.cond_proj_down1(combined_cond).unsqueeze(-1)
        h2 = self.down2(h1) + self.cond_proj_down2(combined_cond).unsqueeze(-1)
        
        # Middle layer
        h_mid = self.mid_conv(h2)
        
        # Upsample with skips
        h_up1 = self.up1(h_mid) + h1
        h_up2 = self.up2(h_up1) + h0
        
        return self.output_conv(h_up2).squeeze(1)


# 2. Diffusion Process -------------------------------------------------------

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
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
    """Generate samples using DDIM sampling"""
    with torch.no_grad():
        # Prepare conditions
        dates = conditions["date"].repeat(num_samples, 1).to(device)
        market_caps = conditions["market_cap"].repeat(num_samples, 1).to(device)
        
        # Initialize noise
        samples = torch.randn(num_samples, 252, device=device)
        
        # Generation process
        for t in tqdm(reversed(range(0, diffusion.num_timesteps, diffusion.num_timesteps // steps)),
                     desc="Generating samples"):
            times = torch.full((num_samples,), t, device=device, dtype=torch.long)
            pred_noise = model(samples, times, dates, market_caps)
            
            # DDIM update
            alpha_bar = diffusion.alpha_bars[t]
            alpha_bar_prev = diffusion.alpha_bars[t-1] if t > 0 else torch.tensor(1.0)
            
            samples = (samples - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            samples = samples * alpha_bar_prev.sqrt() + (1 - alpha_bar_prev).sqrt() * pred_noise
        
        return samples.cpu()


def calculate_metrics(real_data, generated_data):
    """Calculate comparison metrics"""
    metrics = {}
    
    # Basic statistics
    metrics['real_mean'] = real_data.mean().item()
    metrics['gen_mean'] = generated_data.mean().item()
    metrics['real_std'] = real_data.std().item()
    metrics['gen_std'] = generated_data.std().item()
    
    # Correlation
    real_corr = np.corrcoef(real_data.numpy())
    gen_corr = np.corrcoef(generated_data.numpy())
    metrics['real_corr'] = np.abs(real_corr).mean()
    metrics['gen_corr'] = np.abs(gen_corr).mean()
    
    # Distribution similarity
    real_hist = np.histogram(real_data.numpy(), bins=50, density=True)[0]
    gen_hist = np.histogram(generated_data.numpy(), bins=50, density=True)[0]
    metrics['kl_div'] = F.kl_div(
        torch.from_numpy(gen_hist).log(), 
        torch.from_numpy(real_hist), 
        reduction='batchmean'
    ).item()
    
    return metrics


def print_comparison_report(metrics_dict, num_conditions):
    """Print detailed comparison report"""
    print("\n=== Validation Report ===")
    
    # Global statistics
    print("\n[Global Statistics]")
    print(f"{'Metric':<15} | {'Real':>10} | {'Generated':>10} | {'Difference':>10}")
    print("-" * 50)
    for metric in ['mean', 'std', 'corr']:
        real = metrics_dict['global'][f'real_{metric}']
        gen = metrics_dict['global'][f'gen_{metric}']
        diff = abs(real - gen)
        print(f"{metric:<15} | {real:>10.4f} | {gen:>10.4f} | {diff:>10.4f}")
    print(f"KL Divergence: {metrics_dict['global']['kl_div']:.4f}")
    
    # Condition-specific statistics
    print("\n[Condition-Specific Statistics]")
    for i in range(num_conditions):
        cond_metrics = metrics_dict[f'condition_{i}']
        print(f"\nCondition {i+1}:")
        print(f"{'Metric':<15} | {'Real':>10} | {'Generated':>10} | {'Diff %':>10}")
        print("-" * 50)
        for metric in ['mean', 'std']:
            real = cond_metrics[f'real_{metric}']
            gen = cond_metrics[f'gen_{metric}']
            diff_pct = abs(real - gen) / real * 100
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
    for i in range(3):
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
    model = ConditionalUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
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
    print_comparison_report(metrics, num_conditions)
    save_visualizations(real_samples, gen_samples, metrics, output_dir)
    print(f"\nValidation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    run_validation(
        model_path="conditional_diffusion_final.pth",
        data_path="financial_data/sequences/sequences_252.pt"
    )