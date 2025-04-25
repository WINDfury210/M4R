import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # Updated AMP API
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Configuration
sequence_length = 252  # Hard-coded (change to 63 for alternative)
epochs = 10
batch_size = 32
num_timesteps = 200
data_file = f"financial_data/sequences/sequences_{sequence_length}.pt"
output_dir = "financial_outputs"
model_file = os.path.join(output_dir, f"financial_diffusion_{sequence_length}.pth")
os.makedirs(output_dir, exist_ok=True)
time_dim = 512
cond_dim = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, time_dim=512, cond_dim=1):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, "sinusoidal")
        self.cond_embedding = nn.Linear(cond_dim, time_dim)
        self.emb_proj = nn.Linear(time_dim, 64)  # Project emb to d_model
        self.input_proj = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, batch_first=True),  # Enable batch_first
            num_layers=4)
        self.output = nn.Conv1d(64, 1, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, t, cond):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        time_emb = self.time_embedding(t)  # [batch, time_dim]
        cond_emb = self.cond_embedding(cond)  # [batch, time_dim]
        emb = time_emb + cond_emb  # [batch, time_dim]
        emb = self.emb_proj(emb).unsqueeze(1)  # [batch, 1, 64]
        x = self.input_proj(x)  # [batch, 64, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, 64]
        x = x + emb  # Broadcast: [batch, seq_len, 64]
        x = self.transformer(x)  # [batch, seq_len, 64]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [batch, 64, seq_len]
        x = self.output(x).squeeze(1)  # [batch, seq_len]
        return x

# Diffusion process
class Diffusion:
    def __init__(self, num_timesteps, beta_start=0.00005, beta_end=0.01):
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

    def sample(self, model, cond, seq_len, steps, method="ddpm", eta=0.0):
        model.eval()
        with torch.no_grad():
            x = torch.randn(cond.shape[0], seq_len, device=cond.device)
            for i in tqdm(range(steps - 1, -1, -1), desc="Sampling"):
                t = torch.full((cond.shape[0],), i, device=cond.device, dtype=torch.long)
                pred_noise = model(x, t, cond)
                alpha = self.alphas[i]
                alpha_bar = self.alpha_bars[i]
                sigma = eta * torch.sqrt((1 - alpha_bar) / (1 - alpha)) * torch.sqrt(1 - alpha / alpha_bar)
                noise = torch.randn_like(x) if i > 0 else 0
                x = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha) + sigma * noise
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
        return self.sequences[idx], self.conditions[idx]

# Training function
def train(model, diffusion, train_loader, optimizer, epochs, device):
    scaler = GradScaler('cuda')  # Updated API
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sequences, cond in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            sequences = sequences.to(device)
            cond = cond.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (sequences.shape[0],), device=device)
            with autocast('cuda'):  # Updated API
                loss = diffusion.training_loss(model, sequences, t, cond)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), model_file)

# Generation function
def generate(model, diffusion, cond, device, seq_len, steps, method="ddpm", eta=0.0):
    model.load_state_dict(torch.load(model_file))
    return diffusion.sample(model, cond, seq_len, steps, method, eta)

# Main execution
if __name__ == "__main__":
    # Data loader
    train_loader = DataLoader(FinancialDataset(data_file), batch_size=batch_size, 
                              shuffle=True, pin_memory=True)
    
    # Initialize model and diffusion
    model = FinancialDiffusionModel(time_dim=time_dim, cond_dim=cond_dim).to(device)
    diffusion = Diffusion(num_timesteps)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Train
    print(f"Training with sequence length {sequence_length}...")
    train(model, diffusion, train_loader, optimizer, epochs, device)
    
    # Generate samples
    print(f"Generating samples with length {sequence_length}...")
    cond = torch.tensor([[0.2]], device=device).repeat(10, 1)
    samples = generate(model, diffusion, cond, device, seq_len=sequence_length, 
                       steps=500, method="ddpm")
    
    # Save generated samples
    samples_np = samples.cpu().numpy()
    plt.figure(figsize=(10, 5))
    for i in range(min(5, samples_np.shape[0])):
        plt.plot(samples_np[i], label=f"Sample {i+1}")
    plt.xlabel("Day")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"generated_returns_{sequence_length}.png"))
    plt.close()
    torch.save(samples, os.path.join(output_dir, f"generated_returns_{sequence_length}.pt"))
    
    print(f"Generated samples saved to {output_dir}/generated_returns_{sequence_length}.png")