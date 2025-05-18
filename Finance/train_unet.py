import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ===================== 1. 扩散过程 =====================
class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 余弦噪声调度
        self.betas = self._cosine_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def _cosine_beta_schedule(self, s=0.008):
        steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
        f = torch.cos(((steps / self.num_timesteps + s) / (1 + s)) * torch.pi / 2) ** 2
        return torch.clip(1 - f[1:] / f[:-1], 0, 0.999)
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

# ===================== 2. 网络结构 =====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        self.proj = nn.Linear(dim, dim)

    def forward(self, time):
        emb = time.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)

class CondBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None, downsample=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, 
                             stride=2 if downsample else 1, 
                             padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, out_ch) if cond_dim else None

    def forward(self, x, cond=None):
        x = self.conv(x)
        x = self.norm(x)
        if self.cond_proj and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(-1)
        return self.act(x)

class ConditionalUNet(nn.Module):
    def __init__(self, seq_len=252, channels=[32, 64, 128], cond_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.cond_dim = cond_dim
        
        # 条件嵌入
        self.time_embed = TimeEmbedding(cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, cond_dim),  # date(3) + market_cap(1)
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # 编码器
        self.encoder = nn.ModuleList([
            CondBlock(1, channels[0], cond_dim, downsample=True),
            CondBlock(channels[0], channels[1], cond_dim, downsample=True),
            CondBlock(channels[1], channels[2], cond_dim, downsample=False)
        ])
        
        # 中间层
        self.mid_conv1 = CondBlock(channels[-1], channels[-1], cond_dim)
        self.mid_conv2 = CondBlock(channels[-1], channels[-1], cond_dim)
        
        # 解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(channels[2], channels[1], 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, channels[1]),
                nn.SiLU()
            ),
            CondBlock(channels[1]*2, channels[1], cond_dim),
            nn.Sequential(
                nn.ConvTranspose1d(channels[1], channels[0], 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, channels[0]),
                nn.SiLU()
            ),
            CondBlock(channels[0]*2, channels[0], cond_dim)
        ])
        
        # 输出层
        self.final_conv = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x, t, date, market_cap):
        # 条件嵌入
        time_emb = self.time_embed(t)
        cond = torch.cat([date, market_cap], dim=-1)
        cond_emb = self.cond_proj(cond)
        combined_cond = time_emb + cond_emb
        
        # 编码器
        x = x.unsqueeze(1)
        skip1 = self.encoder[0](x, combined_cond)
        skip2 = self.encoder[1](skip1, combined_cond)
        x = self.encoder[2](skip2, combined_cond)
        
        # 中间层
        x = self.mid_conv1(x, combined_cond)
        x = self.mid_conv2(x, combined_cond)
        
        # 解码器
        x = self.decoder[0](x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder[1](x, combined_cond)
        
        x = self.decoder[2](x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder[3](x, combined_cond)
        
        # 输出
        return self.final_conv(x).squeeze(1)

# ===================== 3. 数据加载 =====================
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

# ===================== 4. 训练流程 =====================
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=config["num_timesteps"], device=device)
    model = ConditionalUNet(
        seq_len=config["seq_len"],
        channels=config["channels"],
        cond_dim=config["cond_dim"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    # 数据加载
    dataset = FinancialDataset(config["data_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # 训练循环
    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{config['num_epochs']}"):
            x = batch["sequence"].to(device)
            date = batch["date"].to(device)
            market_cap = batch["market_cap"].to(device)
            
            # 扩散过程
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            noisy_x, noise = diffusion.add_noise(x, t)
            
            # 训练步骤
            optimizer.zero_grad()
            pred_noise = model(noisy_x, t, date, market_cap)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # 打印统计信息
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Loss: {epoch_loss/len(dataloader):.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存模型
        if epoch % config["save_every"] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss/len(dataloader)
            }, os.path.join(config["save_dir"], f"model_epoch_{epoch}.pth"))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "final_model.pth"))
    print(f"Training completed. Models saved to {config['save_dir']}")

# ===================== 5. 生成样本 =====================
def generate_samples(model, diffusion, conditions, num_samples=100, steps=200, device="cuda"):
    model.eval()
    with torch.no_grad():
        # 准备条件数据
        dates = conditions["date"].repeat(num_samples, 1).to(device)
        market_caps = conditions["market_cap"].repeat(num_samples, 1).to(device)
        
        # 初始化噪声
        samples = torch.randn(num_samples, model.seq_len, device=device) * 0.1
        
        # 逐步去噪
        for t in tqdm(reversed(range(0, diffusion.num_timesteps, diffusion.num_timesteps // steps)), 
                      desc="Generating samples"):
            times = torch.full((num_samples,), t, device=device, dtype=torch.long)
            pred_noise = model(samples, times, dates, market_caps)
            
            alpha_bar = diffusion.alpha_bars[t]
            alpha_bar_prev = diffusion.alpha_bars[t-1] if t > 0 else torch.tensor(1.0)
            
            x0_pred = (samples - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
            eps_coef = (1 - alpha_bar_prev) / (1 - alpha_bar)
            eps_coef = torch.clamp(eps_coef, max=2.0)
            
            samples = x0_pred * alpha_bar_prev.sqrt() + \
                     eps_coef.sqrt() * pred_noise * (1 - alpha_bar_prev).sqrt()
        
        return samples.cpu()

# ===================== 主程序 =====================
if __name__ == "__main__":
    config = {
        "num_epochs": 1000,
        "num_timesteps": 2000,
        "batch_size": 64,
        "seq_len": 252,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim": 128,
        "lr": 2e-4,
        "data_path": "financial_data/sequences/sequences_252.pt",
        "save_dir": "saved_models",
        "save_every": 200
    }
    
    # 训练模型
    train_model(config)
    
    # 示例生成
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNet(seq_len=252, channels=[32, 64, 128]).to(device)
    model.load_state_dict(torch.load("saved_models/final_model.pth"))
    
    diffusion = DiffusionProcess(device=device)
    test_conditions = {
        "date": torch.randn(1, 3),
        "market_cap": torch.randn(1, 1)
    }
    generated = generate_samples(model, diffusion, test_conditions)
    print(f"Generated samples shape: {generated.shape}")