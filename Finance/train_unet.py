"""
Diffusion Model Training Script
Train ConditionalUNet1D with MSE, ACF, Std, and Mean losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy import stats
from model import ConditionalUNet1D


# 2. Diffusion Process -------------------------------------------------------

class DiffusionProcess:
    def __init__(self, num_timesteps=2000, device="cpu"):
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
    def __init__(self, data_path, scale_factor=1.0):
        data = torch.load(data_path)
        self.sequences = data["sequences"]
        self.dates = data["start_dates"]
        # Scale data
        self.original_mean = self.sequences.mean().item()
        self.original_std = self.sequences.std().item()
        self.scale_factor = scale_factor
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
        date_diffs = torch.norm(self.dates - annual_date, dim=1)
        closest_indices = torch.argsort(date_diffs)[:num_samples]
        return self.sequences[closest_indices], closest_indices

# 4. Loss Functions -----------------------------------------------------------

def acf_loss(pred, target):
    """Compute MSE loss between ACF of predictions and target using FFT."""
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    pred_acf = torch.fft.irfft(pred_fft * pred_fft.conj(), dim=-1)
    target_acf = torch.fft.irfft(target_fft * target_fft.conj(), dim=-1)
    pred_acf = pred_acf[:, 1:21] / (pred_acf[:, 0:1] + 1e-8)
    target_acf = target_acf[:, 1:21] / (target_acf[:, 0:1] + 1e-8)
    return F.mse_loss(pred_acf, target_acf)

def std_loss(pred, target):
    """Compute MSE loss between standard deviations of predictions and target."""
    pred_std = pred.std(dim=-1)
    target_std = target.std(dim=-1)
    return F.mse_loss(pred_std, target_std)

def mean_loss(pred, target):
    """Compute MSE loss between means of predictions and target."""
    pred_mean = pred.mean(dim=-1)
    target_mean = target.mean(dim=-1)
    return F.mse_loss(pred_mean, target_mean)

def ks_loss(pred, target):
    """Compute KS statistic loss between predictions and target."""
    batch_size = pred.size(0)
    losses = []
    for i in range(batch_size):
        p = pred[i].detach().cpu().numpy()
        t = target[i].detach().cpu().numpy()
        ks_stat, _ = stats.ks_2samp(p, t)
        losses.append(torch.tensor(ks_stat, device=pred.device))
    return torch.stack(losses).mean()

# 5. Training Function --------------------------------------------------------

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNet1D(seq_len=256, channels=config["channels"]).to(device)
    diffusion = DiffusionProcess(num_timesteps=2000, device=device)
    dataset = FinancialDataset(config["data_path"], scale_factor=1.0)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            dates = batch["date"].to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (sequences.size(0),), device=device)
            noisy_x, noise = diffusion.add_noise(sequences, t)
            
            optimizer.zero_grad()
            pred_noise = model(noisy_x, t, dates)
            
            mse_loss = F.mse_loss(pred_noise, noise)
            # acf_loss_val = acf_loss(pred_noise, noise)
            # std_loss_val = std_loss(pred_noise, noise)
            # mean_loss_val = mean_loss(pred_noise, noise)
            ks_loss_val = ks_loss(pred_noise, noise)
            
            loss = mse_loss + ks_loss_val  # Add KS loss with weight 0.5
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.6f}, ")
        
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config["save_dir"], f"model_epoch_{epoch+1}.pth"))
    
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "final_model.pth"))

# 6. Main --------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "data_path": "financial_data/sequences/sequences_256.pt",
        "save_dir": "saved_models",
        "num_epochs": 1000,
        "batch_size": 32,  # Reduced from 64
        "channels": [32, 128, 512, 1024, 2048],
        "lr": 5e-7,  # Reduced from 1e-6
        "save_interval": 500
    }
    os.makedirs(config["save_dir"], exist_ok=True)
    train_model(config)