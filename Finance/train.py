import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration
sequence_length = 252
batch_size = 32
epochs = 100
lr = 1e-4
time_dim = 512
cond_dim = 64
d_model = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_file = f"financial_data/sequences/sequences_{sequence_length}.pt"
output_dir = "financial_outputs"
os.makedirs(output_dir, exist_ok=True)

# Dataset
class FinancialDataset(Dataset):
    def __init__(self, data_file):
        data = torch.load(data_file, weights_only=False)
        self.sequences = data["sequences"].float()  # [N, 252]
        self.conditions = data["conditions"].float()  # [N, 252]
        print(f"Loaded sequences shape: {self.sequences.shape}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.conditions[idx]

# Time embedding
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

# Condition embedding
class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=cond_dim, num_layers=2, 
                         batch_first=True, bidirectional=True)
        self.linear = nn.Linear(cond_dim * 2, cond_dim)
        self.proj = nn.Linear(cond_dim, d_model)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, 
                                             batch_first=True)
    
    def forward(self, cond):
        cond = cond.unsqueeze(-1) if cond.dim() == 2 else cond
        cond_emb, _ = self.gru(cond)
        cond_emb = self.linear(cond_emb)
        cond_emb = self.relu(cond_emb)
        cond_emb = self.proj(cond_emb)
        cond_emb = self.norm(cond_emb)
        cond_res = cond_emb
        cond_emb, _ = self.attention(cond_emb, cond_emb, cond_emb)
        cond_emb = cond_emb + cond_res
        return cond_emb

# Financial diffusion model
class FinancialDiffusionModel(nn.Module):
    def __init__(self, time_dim=512, cond_dim=64, d_model=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.cond_embedding = ConditionEmbedding(cond_dim, d_model)
        self.emb_proj = nn.Linear(time_dim, d_model)
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, 
                                     dim_feedforward=1024, batch_first=True),
            num_layers=10)
        self.output = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=3, padding=1)
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, t, cond):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        time_emb = self.time_embedding(t)
        cond_emb = self.cond_embedding(cond)
        emb = self.emb_proj(time_emb).unsqueeze(1)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = x + emb
        x = x + cond_emb
        x = self.transformer(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.output(x).squeeze(1)
        return x

# Diffusion process
class Diffusion:
    def __init__(self, num_timesteps=100, beta_start=0.00005, beta_end=0.001):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def training_loss(self, model, x0, t, cond):
        noise = torch.randn_like(x0)  # [batch, seq_len]
        xt = self.sqrt_alpha_bars[t][:, None] * x0 + self.sqrt_one_minus_alpha_bars[t][:, None] * noise
        predicted_noise = model(xt, t, cond)
        return F.mse_loss(predicted_noise, noise)
    
    def sample(self, model, cond, seq_len, steps, method="ddim", eta=0.0):
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

# Training function
def train(model, diffusion, train_loader, optimizer, scheduler, epochs, device):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (sequences, cond) in enumerate(train_loader):
            sequences, cond = sequences.to(device), cond.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.num_timesteps, (sequences.shape[0],), device=device)
            loss = diffusion.training_loss(model, sequences, t, cond)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(output_dir, f"financial_diffusion_{sequence_length}.pth"))
    return losses

# Main execution
if __name__ == "__main__":
    print(f"Training with sequence length {sequence_length}...")
    dataset = FinancialDataset(data_file)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = FinancialDiffusionModel(time_dim=time_dim, cond_dim=cond_dim, d_model=d_model).to(device)
    diffusion = Diffusion(num_timesteps=100)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    losses = train(model, diffusion, train_loader, optimizer, scheduler, epochs, device)
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, f"training_loss_{sequence_length}.png"))
    plt.close()
    print(f"Training completed. Model saved to {output_dir}/financial_diffusion_{sequence_length}.pth")