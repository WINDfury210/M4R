import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import os

# Configuration
sequence_length = 252
epochs = 75
batch_size = 32
num_timesteps = 100
data_file = f"financial_data/sequences/sequences_{sequence_length}.pt"
output_dir = "financial_outputs"
model_file = os.path.join(output_dir, f"financial_diffusion_{sequence_length}.pth")
os.makedirs(output_dir, exist_ok=True)
time_dim = 512
cond_dim = 64
d_model = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset statistics
data = torch.load(data_file, weights_only=False)
real_mean = data["sequences"].mean(dim=1, keepdim=True).mean().item()
real_std = data["sequences"].std(dim=1, keepdim=True).mean().item()

# Time embedding module
class TimeEmbedding(nn.Module):
    def __init__(self, dim, type="sinusoidal"):
        super().__init__()
        self.dim = dim
        self.type = type
    
    def forward(self, t):
        if self.type == "sinusoidal":
            half_dim = self.dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
            emb = t[:, None] * emb[None, :]
            return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# Financial diffusion model
class FinancialDiffusionModel(nn.Module):
    def __init__(self, time_dim=512, cond_dim=64, d_model=128):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, "sinusoidal")
        self.cond_embedding = nn.Sequential(
            nn.GRU(input_size=1, hidden_size=cond_dim, num_layers=2, batch_first=True, bidirectional=True),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.ReLU(),
            nn.MultiheadAttention(embed_dim=cond_dim, num_heads=8, batch_first=True),
        )
        self.emb_proj = nn.Linear(time_dim, d_model)
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=8)
        self.output = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=3, padding=1)
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, t, cond):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        time_emb = self.time_embedding(t)  # [batch, time_dim]
        cond_emb, _ = self.cond_embedding[0](cond.unsqueeze(-1))  # [batch, seq_len, cond_dim * 2]
        cond_emb = self.cond_embedding[1](cond_emb)  # [batch, seq_len, cond_dim]
        cond_emb, _ = self.cond_embedding[3]((cond_emb, cond_emb, cond_emb))  # [batch, seq_len, cond_dim]
        emb = self.emb_proj(time_emb).unsqueeze(1)  # [batch, 1, d_model]
        x = self.input_proj(x)  # [batch, d_model, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, d_model]
        x = x + emb  # Broadcast
        x = x + cond_emb  # [batch, seq_len, d_model]
        x_res = x
        x = self.transformer(x)
        x = x + x_res  # Residual
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
        x = self.output(x).squeeze(1)  # [batch, seq_len]
        return x

# Diffusion process (DDIM)
class Diffusion:
    def __init__(self, num_timesteps, beta_start=0.00005, beta_end=0.005):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def training_loss(self, model, x0, t, cond, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.sqrt_alpha_bars[t][:, None] * x0 + self.sqrt_one_minus_alpha_bars[t][:, None] * noise
        pred_noise = model(xt, t, cond)
        loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1])
        return (loss * self.sqrt_one_minus_alpha_bars[t] ** 2).mean()

    def sample(self, model, cond, seq_len, steps, method="ddim", eta=0.0):
        model.eval()
        with torch.no_grad():
            x = torch.randn(cond.shape[0], seq_len, device=cond.device)
            skip = self.num_timesteps // steps
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

# Financial dataset
class FinancialDataset(Dataset):
    def __init__(self, data_file):
        data = torch.load(data_file, weights_only=False)
        self.sequences = data["sequences"]
        self.conditions = data["conditions"]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        scale = torch.rand(1) * 0.2 + 0.9
        seq = seq * scale + torch.randn_like(seq) * 0.01
        return seq, self.conditions[idx]

# Training function
def train(model, diffusion, train_loader, optimizer, scheduler, epochs, device):
    scaler = GradScaler('cuda')
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sequences, cond in train_loader:
            sequences = sequences.to(device)
            cond = cond.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (sequences.shape[0],), device=device)
            with autocast('cuda'):
                loss = diffusion.training_loss(model, sequences, t, cond)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), model_file)

# Generation function
def generate(model, diffusion, cond, device, seq_len, steps, method="ddim", eta=0.0):
    model.load_state_dict(torch.load(model_file))
    samples = diffusion.sample(model, cond, seq_len, steps, method, eta)
    samples = samples * real_std + real_mean
    return samples

# Main execution
if __name__ == "__main__":
    # Data loader
    train_loader = DataLoader(FinancialDataset(data_file), batch_size=batch_size, 
                              shuffle=True, pin_memory=True)
    
    # Initialize model and diffusion
    model = FinancialDiffusionModel(time_dim=time_dim, cond_dim=cond_dim, d_model=d_model).to(device)
    diffusion = Diffusion(num_timesteps)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Train
    print(f"Training with sequence length {sequence_length}...")
    train(model, diffusion, train_loader, optimizer, scheduler, epochs, device)
    print("Training completed.")
    
    # Generate samples
    print(f"Generating samples with length {sequence_length}...")
    cond = torch.ones(10, sequence_length, device=device) * 0.2
    samples = generate(model, diffusion, cond, device, seq_len=sequence_length, 
                       steps=100, method="ddim")
    
    # Save generated samples
    samples_np = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    for i in range(min(5, samples_np.shape[0])):
        plt.plot(samples_np[i], label=f"Sample {i+1}")
    plt.xlabel("Day")
    plt.ylabel("Return")
    plt.ylim(-3 * real_std, 3 * real_std)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"generated_returns_{sequence_length}.png"))
    plt.close()
    torch.save(samples, os.path.join(output_dir, f"generated_returns_{sequence_length}.pt"))
    
    # Evaluation metrics
    print("Evaluation metrics:")
    gen_mean = samples_np.mean()
    gen_std = samples_np.std()
    gen_acf = acf(samples_np[0], nlags=20)
    real_acf = acf(data["sequences"][0].numpy(), nlags=20)
    print(f"Generated Mean: {gen_mean:.6f}, Real Mean: {real_mean:.6f}")
    print(f"Generated Std: {gen_std:.6f}, Real Std: {real_std:.6f}")
    print(f"Generated ACF (lags 1-5): {gen_acf[1:6].tolist()}")
    print(f"Real ACF (lags 1-5): {real_acf[1:6].tolist()}")
    
    # Save ACF plot
    plt.figure(figsize=(8, 4))
    plt.plot(gen_acf, label="Generated")
    plt.plot(real_acf, label="Real")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"acf_comparison_{sequence_length}.png"))
    plt.close()
    
    print(f"Generated samples saved to {output_dir}/generated_returns_{sequence_length}.png")
    print(f"ACF comparison saved to {output_dir}/acf_comparison_{sequence_length}.png")