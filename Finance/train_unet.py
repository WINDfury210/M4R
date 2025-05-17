import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import logging
import numpy as np
from typing import List, Tuple

# ==================== 配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finance_diffusion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 模型核心 ====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        emb = time[:, None] * self.emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class FinancialDiffusion(nn.Module):
    def __init__(
        self,
        seq_len: int = 252,
        cond_dim: int = 4,
        time_dim: int = 256,
        channels: List[int] = [64, 128, 256, 512],
        num_blocks: int = 3,
        activation: str = "silu"
    ):
        super().__init__()
        self.seq_len = seq_len
        
        # 验证序列长度可被下采样次数整除
        n_down = len(channels) - 1
        assert (seq_len % (2 ** n_down)) == 0, \
            f"Sequence length {seq_len} must be divisible by {2 ** n_down} for {n_down} downsamplings"
        
        # 激活函数
        self.act = {
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "relu": nn.ReLU()
        }[activation]

        # 时间编码
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            self.act,
            nn.Linear(time_dim, channels[0])
        )
        
        # 条件编码
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            self.act,
            nn.Linear(time_dim, channels[0])
        )
        
        # 输入层
        self.input_conv = nn.Conv1d(1, channels[0], 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.down_blocks.append(DownBlock(
                in_ch=channels[i],
                out_ch=channels[i+1],
                num_blocks=num_blocks,
                act=self.act
            ))
        
        # 中间层
        self.mid_block = MidBlock(
            channels=channels[-1],
            num_blocks=num_blocks*2,
            act=self.act
        )
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels)-1)):
            self.up_blocks.append(UpBlock(
                in_ch=channels[i+1],
                out_ch=channels[i],
                skip_ch=channels[i],
                num_blocks=num_blocks,
                act=self.act
            ))
        
        # 输出层
        self.output = nn.Conv1d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 保存原始长度
        orig_len = x.shape[-1]
        
        # 时间条件
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(-1)
        
        # 项目条件
        c_emb = self.cond_proj(cond).unsqueeze(-1)
        
        # 初始卷积
        x = self.input_conv(x.unsqueeze(1)) + t_emb + c_emb
        
        # 下采样并保存跳跃连接
        skips = []
        for block in self.down_blocks:
            skips.append(x)
            x = block(x)
        
        # 中间层
        x = self.mid_block(x)
        
        # 上采样并与跳跃连接拼接
        for i, block in enumerate(self.up_blocks):
            x = block(x, skips[-(i+1)])
        
        # 确保输出长度与输入一致
        if x.shape[-1] != orig_len:
            x = F.interpolate(x, size=orig_len, mode='linear')
        
        return self.output(x).squeeze(1)

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, act: nn.Module):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                act
            ])
            in_ch = out_ch
        layers.append(nn.MaxPool1d(2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        skip_ch: int,
        num_blocks: int, 
        act: nn.Module
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, 1)
        
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv1d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                act
            ])
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # 调整skip连接尺寸
        if skip.shape[-1] != x.shape[-1]:
            skip = F.interpolate(skip, size=x.shape[-1], mode='linear')
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.blocks(x)

class MidBlock(nn.Module):
    def __init__(self, channels: int, num_blocks: int, act: nn.Module):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                act
            ])
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ==================== 扩散过程 ====================
class DiffusionProcess(nn.Module):
    def __init__(self, num_timesteps: int = 1000, device: str = "cpu"):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 余弦调度
        t = torch.linspace(0, 1, num_timesteps+1)
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        betas = torch.clip(1 - (f[1:] / f[:-1]), 0, 0.999)
        
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1. - alpha_bars)
        
        # 注册缓冲区
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alpha_bars', alpha_bars.to(device))
        self.register_buffer('sqrt_alpha_bars', sqrt_alpha_bars.to(device))
        self.register_buffer('sqrt_one_minus_alpha_bars', sqrt_one_minus_alpha_bars.to(device))
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        noisy = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return noisy, noise
    
    def sample(self, model: nn.Module, cond: torch.Tensor, steps: int = 100) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            x = torch.randn(cond.shape[0], model.seq_len, device=cond.device)
            
            for t in tqdm(reversed(range(0, self.num_timesteps, self.num_timesteps//steps)), 
                         desc="Sampling"):
                t_batch = torch.full((cond.shape[0],), t, device=cond.device)
                pred_noise = model(x, t_batch, cond)
                
                alpha_bar = self.alpha_bars[t].view(-1, 1)
                alpha_bar_prev = self.alpha_bars[max(t-1, 0)].view(-1, 1)
                
                x = (x - (1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
                    x += sigma * noise
            
            return x

# ==================== 数据加载 ====================
class FinancialDataset(Dataset):
    def __init__(self, data_path: str):
        data = torch.load(data_path)
        self.sequences = data["sequences"].float()
        start_dates = data["start_dates"].float()
        market_caps = data["market_caps"].float()
        self.conditions = torch.cat([start_dates, market_caps], dim=1)
        
        # 标准化
        self.sequence_mean = self.sequences.mean(dim=(0,1), keepdim=True)
        self.sequence_std = self.sequences.std(dim=(0,1), keepdim=True)
        self.sequences = (self.sequences - self.sequence_mean) / (self.sequence_std + 1e-6)
        
        self.cond_mean = self.conditions.mean(dim=0, keepdim=True)
        self.cond_std = self.conditions.std(dim=0, keepdim=True)
        self.conditions = (self.conditions - self.cond_mean) / (self.cond_std + 1e-6)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.conditions[idx]

# ==================== 训练器 ====================
def train_and_evaluate(
    data_path: str,
    save_dir: str = "saved_models",
    epochs: int = 1000,
    batch_size: int = 64,
    model_config: dict = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # 初始化
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据加载
    dataset = FinancialDataset(data_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 模型初始化
    model = FinancialDiffusion(**(model_config or {})).to(device)
    diffusion = DiffusionProcess(num_timesteps=1000, device=device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        
        for x, cond in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, cond = x.to(device), cond.to(device)
            
            # 随机时间步
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            
            # 添加噪声
            noisy_x, noise = diffusion.add_noise(x, t)
            
            # 预测噪声
            pred_noise = model(noisy_x, t, cond)
            
            # 计算损失
            loss = F.mse_loss(pred_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        
        logger.info(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    logger.info("Training completed")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置
    config = {
        "data_path": "financial_data/sequences/sequences_252.pt",
        "save_dir": "saved_models",
        "epochs": 1000,
        "batch_size": 64,
        "model_config": {
            "seq_len": 252,
            "cond_dim": 4,
            "time_dim": 256,
            "channels": [32, 64, 128, 256, 512],  # 控制模型深度和宽度
            "num_blocks": 2,
            "activation": "silu"
        }
    }
    
    # 调整序列长度使其可被下采样次数整除
    n_down = len(config["model_config"]["channels"]) - 1
    if config["model_config"]["seq_len"] % (2 ** n_down) != 0:
        new_len = (config["model_config"]["seq_len"] // (2 ** n_down)) * (2 ** n_down)
        logger.warning(f"Adjusting sequence length from {config['model_config']['seq_len']} to {new_len} for compatibility")
        config["model_config"]["seq_len"] = new_len
    
    try:
        train_and_evaluate(**config)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise