import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 1. 扩散过程 ==========================================================
class DiffusionProcess:
    def __init__(self, num_timesteps: int = 1000, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Cosine噪声调度（比线性调度更稳定）
        self.betas = self._cosine_beta_schedule(num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def _cosine_beta_schedule(self, num_timesteps, s=0.008):
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
        return torch.clip(1 - f[1:] / f[:-1], 0, 0.999)
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

# 2. 时间嵌入 ==========================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        emb = time.float().view(-1, 1) * self.emb.view(1, -1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# 3. 残差块 ============================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 主路径
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        # 条件路径
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
        # 第二层
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 条件注入
        h = h + self.cond_proj(cond).unsqueeze(-1)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.residual(x)

# 4. 注意力层 =========================================================
class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x)
        
        q = self.q(h).view(B, C, L)
        k = self.k(h).view(B, C, L)
        v = self.v(h).view(B, C, L)
        
        attn = torch.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = self.proj_out(out)
        return x + out

# 5. 可配置的Conditional UNet =========================================
class ConditionalUNet(nn.Module):
    def __init__(self, 
                 seq_len: int = 252,
                 base_channels: int = 64,
                 channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
                 num_res_blocks: int = 2,
                 cond_dim: int = 128,
                 use_attention: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.cond_dim = cond_dim
        
        # 时间/条件嵌入
        self.time_embed = TimeEmbedding(cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # 输入层
        self.input_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        current_channels = [base_channels]
        
        # 构建下采样路径
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(current_channels[-1], out_channels, cond_dim)
                ]
                if use_attention and mult in [4, 8]:  # 在高分辨率层添加注意力
                    layers.append(AttentionBlock(out_channels))
                self.down_blocks.append(nn.Sequential(*layers))
                current_channels.append(out_channels)
            
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(out_channels))
                current_channels.append(out_channels)
        
        # 中间层
        self.mid_block1 = ResBlock(current_channels[-1], current_channels[-1], cond_dim)
        self.mid_attn = AttentionBlock(current_channels[-1]) if use_attention else nn.Identity()
        self.mid_block2 = ResBlock(current_channels[-1], current_channels[-1], cond_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        
        # 构建上采样路径
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                layers = [
                    ResBlock(current_channels.pop() + current_channels[-1], out_channels, cond_dim)
                ]
                if use_attention and mult in [4, 8]:
                    layers.append(AttentionBlock(out_channels))
                self.up_blocks.append(nn.Sequential(*layers))
            
            if i != 0:
                self.up_blocks.append(Upsample(out_channels))
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, date: torch.Tensor, market_cap: torch.Tensor) -> torch.Tensor:
        # 条件嵌入
        t_emb = self.time_embed(t)
        cond = torch.cat([date, market_cap], dim=-1)
        c_emb = self.cond_proj(cond)
        combined_cond = t_emb + c_emb
        
        # 输入处理
        x = x.unsqueeze(1)
        h = self.input_conv(x)
        
        # 存储跳连
        skip_connections = [h]
        
        # 下采样
        for block in self.down_blocks:
            h = block(h, combined_cond) if isinstance(block, ResBlock) else block(h)
            skip_connections.append(h)
        
        # 中间层
        h = self.mid_block1(h, combined_cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, combined_cond)
        
        # 上采样
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                h = block(torch.cat([h, skip_connections.pop()], dim=1), combined_cond)
            else:
                h = block(h)
        
        # 输出
        return self.output_conv(h).squeeze(1)

# 6. 下采样/上采样层 ==================================================
class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# 7. 数据加载 ==========================================================
class FinancialDataset(Dataset):
    def __init__(self, data_path: str):
        data = torch.load(data_path)
        self.sequences = data["sequences"]      # [N,252]
        self.dates = data["start_dates"]        # [N,3]
        self.market_caps = data["market_caps"]  # [N,1]
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx],
            "market_cap": self.market_caps[idx]
        }

# 8. 训练函数 ==========================================================
def train(
    num_epochs: int = 500,
    num_timesteps: int = 1000,
    batch_size: int = 32,
    base_channels: int = 64,
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    lr: float = 2e-4,
    data_path: str = "financial_data/sequences/sequences_252.pt",
    save_dir: str = "saved_models",
    save_every: int = 50,
    use_attention: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=num_timesteps, device=device)
    model = ConditionalUNet(
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        use_attention=use_attention
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 数据加载
    dataset = FinancialDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # 学习率调整
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': {
                    'base_channels': base_channels,
                    'channel_mults': channel_mults,
                    'num_res_blocks': num_res_blocks,
                    'cond_dim': model.cond_dim
                }
            }, os.path.join(save_dir, "best_model.pth"))
        
        # 定期保存
        if epoch % save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
        
        # 打印统计信息
        print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print(f"Training completed. Best loss: {best_loss:.6f}")
    print(f"Models saved to {save_dir}")

if __name__ == "__main__":
    train(
        num_epochs=1000,
        num_timesteps=2000,
        batch_size=32,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        lr=2e-4,
        use_attention=True
    )