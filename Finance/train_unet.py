"""
Diffusion Model Training Script
Train ConditionalUNet1D with MSE, ACF, Std, Mean, Corr, Wass, and Skew-Kurt losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy import stats

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
    def __init__(self, channels, num_heads=8):  # Increased heads
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)  # Added dropout

    def forward(self, x):
        batch, ch, seq_len = x.size()
        head_dim = ch // (8 * self.num_heads)
        q = self.query(x).view(batch, self.num_heads, head_dim, seq_len).permute(0, 2, 1, 3)
        k = self.key(x).view(batch, self.num_heads, head_dim, seq_len).permute(0, 2, 3, 1)
        v = self.value(x).view(batch, self.num_heads, ch // self.num_heads, seq_len).permute(0, 2, 0, 3)
        attn = self.softmax(torch.matmul(q, k) / (head_dim ** 0.5))
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).permute(2, 1, 0, 3).contiguous().view(batch, ch, seq_len)
        return x + self.gamma * out

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = out.permute(0, 2, 1)
        out = self.relu(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = out.permute(0, 2, 1)
        out += residual
        return self.relu()

class ConditionalUNet1D(nn.Module):
    def __init__(self, seq_len=256, channels=[64, 128, 256, 512]):  # Aligned channels
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
            self.attentions.append(SelfAttention1D(out_channels, num_heads=8) if 0<i<len(channels) else nn.Identity())
            in_channels = out_channels
        self.mid_conv1 = ResidualBlock1D(channels[-1], channels[-1])
        self.mid_attn = SelfAttention1D(channels[-1], num_heads=8)  # Added mid attention
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
        combined_cond = time_emb + date_emb
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        skips = []
        for i, (conv, res, attn) in enumerate(zip(self.encoder_convs, self.encoder_res, self.attentions)):
            x = F.relu(conv(x))
            x = res(x)
            x = attn(x)
            skips.append(x)
        x = self.mid_conv1(x)
        x = self.mid_attn(x)  # Apply mid attention
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
    def __init__(self, num_timesteps=4000, device="cpu"):  # Increased steps
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self._sigmoid_beta_schedule().to(device)  # Sigmoid schedule
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def _sigmoid_beta_schedule(self):
        t = torch.linspace(0, 1, self.num_timesteps)
        s = 0.015
        betas = 0.0001 + (0.01 - 0.0001) * torch.sigmoid(10 * (t - s))
        return betas
    
    def add_noise(self, x0, t, year=None):
        noise_scale = 0.0005  # Default scale
        if year is not None:
            year_std = {2019: 0.05, 2020: 0.06, 2021: 0.055}.get(year, 0.036374) / 1.0  # Scaled std
            noise_scale = 0.005 * (year_std / 0.06)  # Dynamic scaling
            noise_scale = min(noise_scale, 0.002)  # Cap
        noise = torch.randn_like(x0, device=self.device) * noise_scale
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
        self.sequences = (self.sequences - self.original_mean) / self.original_std * scale_factor
        print(f"Scaled data - Mean: {self.sequences.mean().item():.6f}, Std: {self.sequences.std().item():.6f}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        year = int(self.dates[idx, 0] * (2024 - 2017) + 2017)
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx],
            "year": year
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
    pred_std = pred.std(dim=-1)
    target_std = target.std(dim=-1)
    return F.mse_loss(pred_std, target_std)

def mean_loss(pred, target):
    pred_mean = pred.mean(dim=-1)
    target_mean = target.mean(dim=-1)
    return F.mse_loss(pred_mean, target_mean)

def corr_loss(pred, target):
    batch_size = pred.size(0)
    losses = []
    epsilon = 1e-8
    for i in range(batch_size):
        for lag in range(1, 6):  # Multi-lag
            p_lagged = pred[i, :-lag]
            p_next = pred[i, lag:]
            t_lagged = target[i, :-lag]
            t_next = target[i, lag:]
            p_std = p_lagged.std() * p_next.std() + epsilon
            t_std = t_lagged.std() * t_next.std() + epsilon
            p_corr = ((p_lagged - p_lagged.mean()) * (p_next - p_next.mean())).mean() / p_std
            t_corr = ((t_lagged - t_lagged.mean()) * (t_next - t_next.mean())).mean() / t_std
            if not torch.isnan(p_corr) and not torch.isnan(t_corr):
                losses.append(F.mse_loss(p_corr, t_corr))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)

def wass_loss(pred, target):
    pred_sorted = pred.sort(dim=-1)[0]
    target_sorted = target.sort(dim=-1)[0]
    return F.mse_loss(pred_sorted, target_sorted)

def skew_kurt_loss(pred, target):
    pred_m = pred.mean(dim=-1, keepdim=True)
    target_m = target.mean(dim=-1, keepdim=True)
    pred_s = pred.std(dim=-1, keepdim=True) + 1e-8
    target_s = target.std(dim=-1, keepdim=True) + 1e-8
    pred_skew = ((pred - pred_m) ** 3).mean(dim=-1) / (pred_s ** 3)
    target_skew = ((target - target_m) ** 3).mean(dim=-1) / (target_s ** 3)
    pred_kurt = ((pred - pred_m) ** 4).mean(dim=-1) / (pred_s ** 4) - 3
    target_kurt = ((target - target_m) ** 4).mean(dim=-1) / (target_s ** 4) - 3
    skew_loss = F.mse_loss(pred_skew, target_skew)
    kurt_loss = F.mse_loss(pred_kurt, target_kurt)
    return skew_loss + kurt_loss

def ks_loss(pred, target):
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
    dataset = FinancialDataset(config["data_path"], scale_factor=1.0)  # Scale up
    
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
            years = batch["year"].to(device)
            
            t = torch.randint(0, diffusion.num_timesteps, (sequences.size(0),), device=device)
            noisy_x, noise = diffusion.add_noise(sequences, t, years)
            
            weights = torch.ones(sequences.size(0), device=device)
            
            optimizer.zero_grad()
            pred_noise = model(noisy_x, t, dates)
            
            mse_loss = F.mse_loss(pred_noise, noise)
            # acf_loss_val = acf_loss(pred_noise, noise)
            # std_loss_val = std_loss(pred_noise, noise)
            # mean_loss_val = mean_loss(pred_noise, noise)
            # corr_loss_val = corr_loss(pred_noise, noise)
            # wass_loss_val = wass_loss(pred_noise, noise)
            skew_kurt_loss_val = skew_kurt_loss(pred_noise, noise)
            # ks_loss_val = ks_loss(pred_noise, noise)
            
            loss = (mse_loss + 5.0 * skew_kurt_loss_val)
            loss = loss * weights.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.6f}, "
              f"MSE: {mse_loss.item():.6f}",
              f"Skew-Kurt: {skew_kurt_loss_val.item():.6f}")
        
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
        "num_epochs": 2000,  # Increased for convergence
        "batch_size": 64,
        "channels": [64, 128, 256, 512, 1024],  # Aligned with model
        "lr": 1e-5,  # Adjusted for stability
        "save_interval": 500
    }
    os.makedirs(config["save_dir"], exist_ok=True)
    train_model(config)