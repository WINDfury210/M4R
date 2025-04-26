import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import ks_2samp, skew, kurtosis
from scipy.stats import wasserstein_distance
import os

# Configuration
sequence_length = 252
num_samples = 10
steps = 500
data_file = f"financial_data/sequences/sequences_{sequence_length}.pt"
model_file = f"financial_outputs/financial_diffusion_{sequence_length}.pth"
output_dir = "financial_outputs"
metrics_file = os.path.join(output_dir, f"metrics_{sequence_length}.txt")
os.makedirs(output_dir, exist_ok=True)
time_dim = 512
cond_dim = 64
d_model = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = torch.load(data_file, weights_only=False)
real_mean = data["sequences"].mean().item()
real_std = data["sequences"].std().item()
print(f"Loaded sequences shape: {data['sequences'].shape}")

# Time embedding module
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# Condition embedding module
class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.gru = nn.GRU(1, cond_dim, num_layers=2, batch_first=True)
        self.proj = nn.Linear(cond_dim, d_model)
        self.cond_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, cond):
        cond = cond.unsqueeze(-1) if cond.dim() == 2 else cond
        cond_emb, _ = self.gru(cond)
        cond_emb = self.proj(cond_emb)
        cond_emb = self.relu(cond_emb)
        cond_emb, _ = self.cond_attn(cond_emb, cond_emb, cond_emb)
        cond_emb = self.norm(cond_emb)
        return cond_emb

# Financial diffusion model
class FinancialDiffusionModel(nn.Module):
    def __init__(self, time_dim=512, cond_dim=64, d_model=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.cond_embedding = ConditionEmbedding(cond_dim, d_model)
        self.input_proj = nn.Linear(1, d_model)
        self.emb_proj = nn.Linear(time_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, 
                                     dim_feedforward=1024, batch_first=True),
            num_layers=8)
        self.output = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, t, cond):
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        time_emb = self.time_embedding(t)  # [batch, time_dim]
        cond_emb = self.cond_embedding(cond)  # [batch, seq_len, d_model]
        emb = self.emb_proj(time_emb).unsqueeze(1)  # [batch, 1, d_model]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = x + emb
        x = x + cond_emb
        x = self.transformer(x)
        x = self.norm(x)
        x = self.output(x).squeeze(-1)  # [batch, seq_len]
        return x

# Diffusion process
class Diffusion:
    def __init__(self, num_timesteps=200, beta_start=0.00005, beta_end=0.005):
        self.num_timesteps = num_timesteps
        self.betas = torch.cos(torch.linspace(0, np.pi/2, num_timesteps)) * (beta_end - beta_start) + beta_start
        self.betas = self.betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def sample(self, model, cond, seq_len, steps, method="ddim", eta=0.1):
        model.eval()
        with torch.no_grad():
            x = torch.randn(cond.shape[0], seq_len, device=cond.device)
            steps = min(steps, self.num_timesteps)
            skip = max(1, self.num_timesteps // steps)
            timesteps = torch.arange(self.num_timesteps - 1, -1, -skip, device=device)
            for i, t_idx in enumerate(timesteps):
                t = torch.full((cond.shape[0],), t_idx, device=cond.device, dtype=torch.long)
                pred_noise = model(x, t, cond)
                alpha_bar = self.alpha_bars[t_idx]
                alpha_bar_prev = self.alpha_bars[max(t_idx - skip, 0)]
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                x = (x - (1 - alpha_bar) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
                if method == "ddim":
                    x = torch.sqrt(alpha_bar_prev) * x + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_noise
                else:
                    x = x / torch.sqrt(self.alphas[t_idx])
                noise = torch.randn_like(x) if i < len(timesteps) - 1 else 0
                x = x + sigma * noise
        return x

# Energy distance implementation
def energy_distance(x, y):
    x = x.flatten()
    y = y.flatten()
    n, m = len(x), len(y)
    xx = np.mean(np.abs(x[:, None] - x[None, :]))
    yy = np.mean(np.abs(y[:, None] - y[None, :]))
    xy = np.mean(np.abs(x[:, None] - y[None, :]))
    return np.sqrt(2 * xy - xx - yy)

# Generate samples
def generate_samples(model, diffusion, cond, seq_len, steps, method="ddim", eta=0.1):
    model.load_state_dict(torch.load(model_file, map_location=device))
    samples = diffusion.sample(model, cond, seq_len, steps, method, eta)
    samples = samples * real_std + real_mean
    return samples

# Evaluate samples
def evaluate_samples(samples, real_data, nlags=20, window=20):
    samples_np = samples.cpu().numpy()
    real_np = real_data[:len(samples_np)].cpu().numpy()
    
    # Metrics
    metrics = {}
    metrics["gen_mean"] = samples_np.mean()
    metrics["real_mean"] = real_np.mean()
    metrics["gen_std"] = samples_np.std()
    metrics["real_std"] = real_np.std()
    metrics["gen_max"] = samples_np.max()
    metrics["gen_min"] = samples_np.min()
    metrics["real_max"] = real_np.max()
    metrics["real_min"] = real_np.min()
    metrics["gen_acf"] = acf(samples_np[0], nlags=nlags)
    metrics["real_acf"] = acf(real_np[0], nlags=nlags)
    metrics["gen_skew"] = skew(samples_np.flatten())
    metrics["real_skew"] = skew(real_np.flatten())
    metrics["gen_kurt"] = kurtosis(samples_np.flatten(), fisher=True)
    metrics["real_kurt"] = kurtosis(real_np.flatten(), fisher=True)
    
    # Volatility clustering
    gen_vol = np.array([np.std(samples_np[:, i:i+window]) for i in range(samples_np.shape[1] - window)])
    real_vol = np.array([np.std(real_np[:, i:i+window]) for i in range(real_np.shape[1] - window)])
    metrics["gen_vol_mean"] = gen_vol.mean()
    metrics["real_vol_mean"] = real_vol.mean()
    metrics["gen_vol_std"] = gen_vol.std()
    metrics["real_vol_std"] = real_vol.std()
    
    # Same distribution metrics
    ks_stat, ks_pval = ks_2samp(samples_np.flatten(), real_np.flatten())
    metrics["ks_stat"] = ks_stat
    metrics["ks_pval"] = ks_pval
    metrics["wasserstein"] = wasserstein_distance(samples_np.flatten(), real_np.flatten())
    metrics["energy_dist"] = energy_distance(samples_np, real_np)
    metrics["acf_mse"] = np.mean((metrics["gen_acf"] - metrics["real_acf"]) ** 2)
    metrics["vol_mse"] = np.mean((gen_vol - real_vol) ** 2)
    
    return metrics

# Save and display metrics
def save_metrics(metrics, file_path):
    with open(file_path, 'w') as f:
        print("Evaluation Metrics:")
        f.write("Evaluation Metrics:\n")
        
        print(f"Generated Mean: {metrics['gen_mean']:.6f}, Real Mean: {metrics['real_mean']:.6f}")
        f.write(f"Generated Mean: {metrics['gen_mean']:.6f}, Real Mean: {metrics['real_mean']:.6f}\n")
        
        print(f"Generated Std: {metrics['gen_std']:.6f}, Real Std: {metrics['real_std']:.6f}")
        f.write(f"Generated Std: {metrics['gen_std']:.6f}, Real Std: {metrics['real_std']:.6f}\n")
        
        print(f"Generated Max: {metrics['gen_max']:.6f}, Real Max: {metrics['real_max']:.6f}")
        f.write(f"Generated Max: {metrics['gen_max']:.6f}, Real Max: {metrics['real_max']:.6f}\n")
        
        print(f"Generated Min: {metrics['gen_min']:.6f}, Real Min: {metrics['real_min']:.6f}")
        f.write(f"Generated Min: {metrics['gen_min']:.6f}, Real Min: {metrics['real_min']:.6f}\n")
        
        print(f"Generated Skewness: {metrics['gen_skew']:.6f}, Real Skewness: {metrics['real_skew']:.6f}")
        f.write(f"Generated Skewness: {metrics['gen_skew']:.6f}, Real Skewness: {metrics['real_skew']:.6f}\n")
        
        print(f"Generated Kurtosis: {metrics['gen_kurt']:.6f}, Real Kurtosis: {metrics['real_kurt']:.6f}")
        f.write(f"Generated Kurtosis: {metrics['gen_kurt']:.6f}, Real Kurtosis: {metrics['real_kurt']:.6f}\n")
        
        print(f"Generated Volatility Mean: {metrics['gen_vol_mean']:.6f}, Real Volatility Mean: {metrics['real_vol_mean']:.6f}")
        f.write(f"Generated Volatility Mean: {metrics['gen_vol_mean']:.6f}, Real Volatility Mean: {metrics['real_vol_mean']:.6f}\n")
        
        print(f"Generated Volatility Std: {metrics['gen_vol_std']:.6f}, Real Volatility Std: {metrics['real_vol_std']:.6f}")
        f.write(f"Generated Volatility Std: {metrics['gen_vol_std']:.6f}, Real Volatility Std: {metrics['real_vol_std']:.6f}\n")
        
        print(f"KS Statistic: {metrics['ks_stat']:.6f}, KS p-value: {metrics['ks_pval']:.6f}")
        f.write(f"KS Statistic: {metrics['ks_stat']:.6f}, KS p-value: {metrics['ks_pval']:.6f}\n")
        
        print(f"Wasserstein Distance: {metrics['wasserstein']:.6f}")
        f.write(f"Wasserstein Distance: {metrics['wasserstein']:.6f}\n")
        
        print(f"Energy Distance: {metrics['energy_dist']:.6f}")
        f.write(f"Energy Distance: {metrics['energy_dist']:.6f}\n")
        
        print(f"ACF MSE: {metrics['acf_mse']:.6f}")
        f.write(f"ACF MSE: {metrics['acf_mse']:.6f}\n")
        
        print(f"Volatility MSE: {metrics['vol_mse']:.6f}")
        f.write(f"Volatility MSE: {metrics['vol_mse']:.6f}\n")
        
        print("Generated ACF (lags 1-5):", metrics["gen_acf"][1:6].tolist())
        f.write(f"Generated ACF (lags 1-5): {metrics['gen_acf'][1:6].tolist()}\n")
        
        print("Real ACF (lags 1-5):", metrics["real_acf"][1:6].tolist())
        f.write(f"Real ACF (lags 1-5): {metrics['real_acf'][1:6].tolist()}\n")
        
        print("Generated ACF (lags 1-20):", metrics["gen_acf"][1:].tolist())
        f.write(f"Generated ACF (lags 1-20): {metrics['gen_acf'][1:].tolist()}\n")
        
        print("Real ACF (lags 1-20):", metrics["real_acf"][1:].tolist())
        f.write(f"Real ACF (lags 1-20): {metrics['real_acf'][1:].tolist()}\n")

# Main execution
if __name__ == "__main__":
    # Initialize model and diffusion
    model = FinancialDiffusionModel(time_dim=time_dim, cond_dim=cond_dim, d_model=d_model).to(device)
    diffusion = Diffusion(num_timesteps=200)
    
    # Generate samples
    print(f"Generating {num_samples} samples with length {sequence_length}...")
    cond = data["conditions"][:num_samples].to(device)
    samples = generate_samples(model, diffusion, cond, sequence_length, steps)
    
    # Save generated samples
    samples_np = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    for i in range(min(5, samples_np.shape[0])):
        plt.plot(samples_np[i], label=f"Sample {i+1}")
    plt.xlabel("Day")
    plt.ylabel("Return")
    plt.ylim(-0.15, 0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"generated_returns_{sequence_length}.png"))
    plt.close()
    torch.save(samples, os.path.join(output_dir, f"generated_returns_{sequence_length}.pt"))
    
    # Evaluate samples
    metrics = evaluate_samples(samples, data["sequences"], nlags=20, window=20)
    save_metrics(metrics, metrics_file)
    
    # Save ACF plot
    plt.figure(figsize=(8, 4))
    plt.plot(metrics["gen_acf"], label="Generated")
    plt.plot(metrics["real_acf"], label="Real")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"acf_comparison_{sequence_length}.png"))
    plt.close()
    
    print(f"Generated samples saved to {output_dir}/generated_returns_{sequence_length}.png")
    print(f"ACF comparison saved to {output_dir}/acf_comparison_{sequence_length}.png")
    print(f"Metrics saved to {metrics_file}")