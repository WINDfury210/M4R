import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt


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
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# Condition embedding
class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.gru = nn.GRU(1, cond_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(cond_dim, d_model)
        self.cond_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, cond):
        cond = cond.unsqueeze(-1) if cond.dim() == 2 else cond
        cond_emb, _ = self.gru(cond)
        cond_emb = self.proj(cond_emb)
        cond_emb = self.relu(cond_emb)
        cond_emb, _ = self.cond_attn(cond_emb, cond_emb, cond_emb)
        cond_emb = self.norm(cond_emb)
        cond_emb = self.dropout(cond_emb)
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
        self.dropout = nn.Dropout(0.3)
    
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
        x = self.dropout(x)
        x = self.output(x).squeeze(-1)  # [batch, seq_len]
        return x

# Diffusion process
class Diffusion:
    def __init__(self, num_timesteps=200, beta_start=0.00005, beta_end=0.002):
        self.num_timesteps = num_timesteps
        self.betas = torch.cos(torch.linspace(0, np.pi/2, num_timesteps)) * (beta_end - beta_start) + beta_start
        self.betas = self.betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def training_loss(self, model, x0, t, cond):
        noise = torch.randn_like(x0)  # [batch, seq_len]
        xt = self.sqrt_alpha_bars[t][:, None] * x0 + self.sqrt_one_minus_alpha_bars[t][:, None] * noise
        predicted_noise = model(xt, t, cond)
        mse_loss = F.mse_loss(predicted_noise, noise)
        
        return mse_loss
        # # Encourage negative skew and heavy tails
        
        # skew_penalty = torch.mean(torch.relu(predicted_noise)) * 0.01
        # kurt_penalty = -torch.mean(predicted_noise**4) * 0.01
        
        # # Distribution alignment
        # gen_dist = predicted_noise.flatten()
        # real_dist = noise.flatten()
        # adv_loss = torch.mean((gen_dist - real_dist)**2) * 0.001
        # return mse_loss + skew_penalty + kurt_penalty + adv_loss
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(output_dir, f"financial_diffusion_{sequence_length}.pth"))
    return losses

# Main execution
if __name__ == "__main__":
    # Configuration
    sequence_length = 252
    batch_size = 256
    epochs = 200
    lr = 1e-5
    time_dim = 512
    cond_dim = 64
    d_model = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file = f"financial_data/sequences/sequences_{sequence_length}.pt"
    output_dir = "financial_outputs"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Training with sequence length {sequence_length}...")
    dataset = FinancialDataset(data_file)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = FinancialDiffusionModel(time_dim=time_dim, cond_dim=cond_dim, d_model=d_model).to(device)
    diffusion = Diffusion(num_timesteps=200)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    losses = train(model, diffusion, train_loader, optimizer, scheduler, epochs, device)
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, f"training_loss_{sequence_length}.png"))
    plt.close()
    print(f"Training completed. Model saved to {output_dir}/financial_diffusion_{sequence_length}.pth")