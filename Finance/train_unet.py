import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import math

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
        f = torch.cos(((steps / self.num_timesteps + s) / (1 + s)) * math.pi / 2) ** 2
        return torch.clip(1 - f[1:] / f[:-1], 0, 0.999)
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

# ===================== 2. 严格对称的UNet =====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
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
        if self.cond_proj is not None and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(-1)
        return self.act(x)

class SymmetricUNet(nn.Module):
    def __init__(self, seq_len=252, channels=[32, 64, 128, 256], cond_dim=128):
        super().__init__()
        # 修改为支持252长度的4层结构：252 → 126 → 63 → 32 → 16 → 32 → 63 → 126 → 252
        self.seq_len = seq_len
        self.channels = channels
        self.num_levels = len(channels)
        
        # 条件系统
        self.time_embed = TimeEmbedding(cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # 输入层
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        
        # 编码器（全部下采样）
        self.encoder = nn.ModuleList()
        for i in range(self.num_levels):
            in_ch = channels[i-1] if i > 0 else channels[0]
            self.encoder.append(
                CondBlock(in_ch, channels[i], 
                         cond_dim if i==self.num_levels-1 else None,
                         downsample=True)
            )
        
        # 中间层
        self.mid_conv1 = CondBlock(channels[-1], channels[-1], None)
        self.mid_conv2 = CondBlock(channels[-1], channels[-1], None)
        
        # 解码器（严格对称上采样）
        self.decoder = nn.ModuleList()
        for i in reversed(range(self.num_levels)):
            # 上采样层
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose1d(
                    channels[i], channels[max(i-1,0)],
                    kernel_size=3, stride=2,
                    padding=1, output_padding=1
                ),
                nn.GroupNorm(8, channels[max(i-1,0)]),
                nn.SiLU()
            ))
            # 条件块
            use_cond = (i == self.num_levels-1)
            self.decoder.append(
                CondBlock(channels[max(i-1,0)]*2, channels[max(i-1,0)],
                         cond_dim if use_cond else None)
            )
        
        # 最终补偿层（处理252→126→63→32→16→32→63→126→251的1点差异）
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(channels[0], channels[0], 
                             kernel_size=3, stride=2,
                             padding=1, output_padding=0),
            nn.GroupNorm(8, channels[0]),
            nn.SiLU()
        )
        self.final_conv = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x, t, date, market_cap):
        # 条件嵌入
        time_emb = self.time_embed(t)
        cond = torch.cat([date, market_cap], dim=-1)
        cond_emb = self.cond_proj(cond)
        combined_cond = time_emb + cond_emb
        
        # 输入处理
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        skips = [x]
        
        # 编码器
        for i, block in enumerate(self.encoder):
            x = block(x, combined_cond if i==self.num_levels-1 else None)
            skips.append(x)
        
        # 中间层
        x = self.mid_conv1(x, None)
        x = self.mid_conv2(x, None)
        
        # 解码器
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1]-x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            use_cond = (i == len(self.decoder)-2)
            x = self.decoder[i+1](x, combined_cond if use_cond else None)
        
        # 最终补偿
        x = self.final_upsample(x)
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear')
        return self.final_conv(x).squeeze(1)

# ===================== 3. 数据管道 =====================
class FinancialDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.sequences = data["sequences"]  # [N, 252]
        self.dates = data["start_dates"]    # [N, 3]
        self.market_caps = data["market_caps"] # [N, 1]
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx],
            "market_cap": self.market_caps[idx]
        }

# ===================== 4. 训练系统 =====================
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=config["num_timesteps"], device=device)
    model = SymmetricUNet(
        seq_len=config["seq_len"],
        channels=config["channels"],
        cond_dim=config["cond_dim"]
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    # 数据加载
    dataset = FinancialDataset(config["data_path"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    
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
        
        # 日志记录
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Loss: {epoch_loss/len(dataloader):.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if epoch % config["save_every"] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss/len(dataloader)
            }, os.path.join(config["save_dir"], f"model_epoch_{epoch}.pth"))
    
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "final_model.pth"))

# ===================== 5. 主程序 =====================
if __name__ == "__main__":
    # 配置（严格适配252长度的4层结构）
    config = {
        "num_epochs": 1000,
        "num_timesteps": 2000,
        "batch_size": 64,
        "seq_len": 252,  # 严格保持252
        "channels": [32, 64, 128, 256],  # 4层结构
        "cond_dim": 128,
        "lr": 2e-4,
        "data_path": "financial_data/sequences_252.pt",
        "save_dir": "saved_models",
        "save_every": 200
    }
    
    # 验证模型结构
    print("验证模型结构...")
    test_model = SymmetricUNet(seq_len=252, channels=[32, 64, 128, 256])
    test_input = torch.randn(2, 252)
    test_t = torch.randint(0, 1000, (2,))
    test_date = torch.randn(2, 3)
    test_mcap = torch.randn(2, 1)
    output = test_model(test_input, test_t, test_date, test_mcap)
    print(f"测试通过！输入: {test_input.shape} -> 输出: {output.shape}")
    assert output.shape == test_input.shape
    
    # 开始训练
    print("\n开始训练...")
    train_model(config)