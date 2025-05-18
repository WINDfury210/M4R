import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# 1. 扩散过程 ==========================================================
class DiffusionProcess:
    def __init__(self, num_timesteps: int = 1000, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 线性噪声调度
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
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
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1)
        )
        
        # 条件路径
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels)
        )
        
        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
        
        # 第二层
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.cond_proj(cond.squeeze(-1)).unsqueeze(-1)
        h = self.block2(h)
        return h + self.residual(x)

# 4. 下采样/上采样层 ==================================================
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

# 5. 可配置的Conditional UNet =========================================
class ConditionalUNet(nn.Module):
    def __init__(self, 
                 seq_len: int = 252,
                 base_channels: int = 32,
                 channel_mults: Tuple[int, ...] = (1, 2, 4),
                 num_res_blocks: int = 2,
                 cond_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.cond_dim = cond_dim
        
        # 时间/条件嵌入
        self.time_embed = TimeEmbedding(cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, cond_dim),  # date(3) + market_cap(1)
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.Dropout(dropout)
        )
        
        # 输入层 [B,1,seq_len]
        self.input_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_condition_projs = nn.ModuleList()
        
        in_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            
            # 每层包含多个残差块
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_channels, out_channels, cond_dim))
                in_channels = out_channels
            
            self.down_blocks.append(blocks)
            self.down_condition_projs.append(nn.Linear(cond_dim, out_channels))
            
            # 如果不是最后一层，添加下采样
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(out_channels))
        
        # 中间层
        self.mid_blocks = nn.ModuleList([
            ResBlock(in_channels, in_channels, cond_dim),
            ResBlock(in_channels, in_channels, cond_dim)
        ])
        self.mid_cond_proj = nn.Linear(cond_dim, in_channels)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_condition_projs = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            
            # 每层包含多个残差块
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(in_channels + out_channels, out_channels, cond_dim))
                self.up_condition_projs.append(nn.Linear(cond_dim, out_channels))
                in_channels = out_channels
            
            # 如果不是最后一层，添加上采样
            if i != 0:
                self.up_blocks.append(Upsample(out_channels))
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, date: torch.Tensor, market_cap: torch.Tensor) -> torch.Tensor:
        # 条件嵌入
        t_emb = self.time_embed(t)
        cond = torch.cat([date, market_cap], dim=-1)
        c_emb = self.cond_proj(cond)
        combined_cond = t_emb + c_emb
        
        # 输入处理
        x = x.unsqueeze(1)  # [B,1,seq_len]
        h = self.input_conv(x)
        
        # 存储下采样各层的特征用于跳连
        skip_connections = []
        cond_proj_idx = 0
        
        # 下采样路径
        for block in self.down_blocks:
            if isinstance(block, nn.ModuleList):  # 残差块组
                for res_block in block:
                    cond_proj = self.down_condition_projs[cond_proj_idx]
                    h = res_block(h, cond_proj(combined_cond).unsqueeze(-1))
                cond_proj_idx += 1
                skip_connections.append(h)
            else:  # 下采样层
                h = block(h)
        
        # 中间层
        for block in self.mid_blocks:
            h = block(h, self.mid_cond_proj(combined_cond).unsqueeze(-1))
        
        # 上采样路径
        for block in self.up_blocks:
            if isinstance(block, ResBlock):  # 残差块
                cond_proj = self.up_condition_projs.pop(0)
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)  # 跳连
                h = block(h, cond_proj(combined_cond).unsqueeze(-1))
            else:  # 上采样层
                h = block(h)
        
        # 输出处理
        out = self.output_conv(h)
        return out.squeeze(1)  # [B,seq_len]

# 6. 数据加载 ==========================================================
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

# 7. 训练函数 ==========================================================
def train(
    num_epochs: int = 500,
    num_timesteps: int = 1000,
    batch_size: int = 32,
    base_channels: int = 64,
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    lr: float = 2e-4,
    dropout: float = 0.1,
    data_path: str = "financial_data/sequences/sequences_252.pt",
    save_path: str = "conditional_diffusion_final.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=num_timesteps, device=device)
    model = ConditionalUNet(
        seq_len=252,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        dropout=dropout
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
        
        for batch in dataloader:
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
                'config': {
                    'base_channels': base_channels,
                    'channel_mults': channel_mults,
                    'num_res_blocks': num_res_blocks,
                    'cond_dim': model.cond_dim
                }
            }, save_path)
        
        # 打印统计信息
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print(f"Training completed. Best loss: {best_loss:.6f}")
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # 示例配置
    train(
        num_epochs=500,
        num_timesteps=2000,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),  # 4层UNet
        num_res_blocks=2,
        lr=2e-4,
        dropout=0.1
    )