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

# ===================== 2. 网络组件 =====================
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
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, ch, seq_len = x.size()
        q = self.query(x).view(batch, -1, seq_len).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, seq_len)
        v = self.value(x).view(batch, -1, seq_len)
        attn = self.softmax(torch.bmm(q, k) / (ch // 8) ** 0.5)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch, ch, seq_len)
        return x + self.gamma * out

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# ===================== 3. 条件UNet =====================
class ConditionalUNet1D(nn.Module):
    def __init__(self, seq_len=252, channels=[32, 64, 128, 256], cond_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_levels = len(channels)
        
        # 时间/条件嵌入
        self.time_embed = TimeEmbedding(cond_dim, embedding_type="sinusoidal")
        self.cond_proj = nn.Sequential(
            nn.Linear(4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # 输入层
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        
        # 编码器
        self.encoder_convs = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_channels = channels[0]
        for i, out_channels in enumerate(channels):
            self.encoder_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                              stride=2 if i>0 else 1, padding=1))
            self.encoder_res.append(ResidualBlock1D(out_channels, out_channels))
            self.attentions.append(SelfAttention1D(out_channels) if i in [1, 2, 3] else nn.Identity())
            in_channels = out_channels
        
        # 中间层
        self.mid_conv1 = ResidualBlock1D(channels[-1], channels[-1])
        self.mid_conv2 = ResidualBlock1D(channels[-1], channels[-1])
        
        # 解码器
        self.decoder_convs = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        for i in range(len(channels)-1):
            in_channels = channels[-1-i] + channels[-2-i]
            out_channels = channels[-2-i]
            self.decoder_convs.append(nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ))
            self.decoder_res.append(ResidualBlock1D(out_channels, out_channels))
        
        # 输出层
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(channels[0], channels[0],
                              kernel_size=3, stride=2,
                              padding=1, output_padding=0),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU()
        )
        self.final_conv = nn.Conv1d(channels[0], 1, kernel_size=1)
        
        # 条件投影
        self.fc_cond = nn.Linear(cond_dim, channels[-1])

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
        for conv, res, attn in zip(self.encoder_convs, self.encoder_res, self.attentions):
            x = F.relu(conv(x))
            x = res(x)
            x = attn(x)
            skips.append(x)
        
        # 中间层+条件注入
        x = self.mid_conv1(x)
        x = x + self.fc_cond(combined_cond).unsqueeze(-1)
        x = self.mid_conv2(x)
        
        # 解码器
        for conv, res in zip(self.decoder_convs, self.decoder_res):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = F.relu(conv(x))
            x = res(x)
        
        # 最终输出
        x = torch.cat([x, skips[0]], dim=1)
        x = self.final_upsample(x)
        if x.shape[-1] != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode='linear')
        return self.final_conv(x).squeeze(1)

# ===================== 4. 数据管道 =====================
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

# ===================== 5. 训练系统 =====================
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=config["num_timesteps"], device=device)
    model = ConditionalUNet1D(
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

# ===================== 6. 主程序 =====================
if __name__ == "__main__":
    # 配置
    config = {
        "num_epochs": 1000,
        "num_timesteps": 2000,
        "batch_size": 64,
        "seq_len": 252,
        "channels": [32, 64, 128, 256],
        "cond_dim": 128,
        "lr": 2e-4,
        "data_path": "financial_data/sequences_252.pt",
        "save_dir": "saved_models",
        "save_every": 200
    }
    
    # 验证模型
    print("验证模型结构...")
    test_model = ConditionalUNet1D(seq_len=252, channels=[32, 64, 128, 256])
    test_input = torch.randn(2, 252)
    test_t = torch.randint(0, 1000, (2,))
    test_date = torch.randn(2, 3)
    test_mcap = torch.randn(2, 1)
    output = test_model(test_input, test_t, test_date, test_mcap)
    print(f"测试通过！输入: {test_input.shape} -> 输出: {output.shape}")
    assert output.shape == (2, 252)
    
    # 开始训练
    print("\n开始训练...")
    train_model(config)