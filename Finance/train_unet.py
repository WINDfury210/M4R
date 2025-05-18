import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 优化器在这里导入
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# 1. 时间嵌入模块 ======================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """输入: [batch_size], 输出: [batch_size, dim]"""
        emb = time.float().view(-1, 1) * self.emb.view(1, -1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# 2. 条件投影模块 ======================================================
class ConditionProjection(nn.Module):
    def __init__(self, cond_dim: int = 4, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, date: torch.Tensor, market_cap: torch.Tensor) -> torch.Tensor:
        """输入: date [B,3], market_cap [B,1], 输出: [B,proj_dim]"""
        cond = torch.cat([date, market_cap], dim=-1)
        return self.proj(cond)

# 3. 残差块 (1D版本) ==================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 条件注入
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.SiLU()
        )
        
        # 捷径连接
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1) 
            if in_channels != out_channels 
            else nn.Identity()
        )
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """输入: x [B,C,T], t_emb [B,time_dim], 输出: [B,out_channels,T]"""
        residual = self.shortcut(x)
        
        # 条件注入 [B,out_channels,1]
        t = self.time_proj(t_emb).unsqueeze(-1)
        
        # 主路径处理
        x = F.silu(self.bn1(self.conv1(x)) + t)
        x = self.bn2(self.conv2(x))
        
        return F.silu(x + residual)

# 4. 下采样块 =========================================================
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.res_block = ResidualBlock1D(in_channels, out_channels, time_dim)
        self.downsample = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=3, stride=2, padding=1
        )
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """输入: x [B,in_channels,T], 输出: [B,out_channels,T//2]"""
        x = self.res_block(x, t_emb)
        return self.downsample(x)

# 5. 上采样块 =========================================================
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, in_channels, 
            kernel_size=3, stride=2, 
            padding=1, output_padding=1
        )
        self.res_block = ResidualBlock1D(in_channels*2, out_channels, time_dim)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """输入: x [B,in_channels,T], skip [B,out_channels,T*2], 输出: [B,out_channels,T*2]"""
        x = self.upsample(x)
        
        # 确保尺寸匹配
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear')
            
        x = torch.cat([x, skip], dim=1)
        return self.res_block(x, t_emb)

# 6. 条件UNet模型 =====================================================
class ConditionalUNet1D(nn.Module):
    def __init__(
        self, 
        input_dim: int = 1,
        output_dim: int = 1,
        time_dim: int = 128,
        channels: List[int] = [32, 64, 128]
    ):
        super().__init__()
        # 时间/条件嵌入
        self.time_embed = TimeEmbedding(time_dim)
        self.cond_proj = ConditionProjection(cond_dim=4, proj_dim=time_dim)
        
        # 输入层 [B,1,252] -> [B,32,252]
        self.input_conv = nn.Conv1d(input_dim, channels[0], kernel_size=3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.down_blocks.append(
                DownBlock(channels[i], channels[i+1], time_dim)
            )
        
        # 中间层 [B,128,63] -> [B,128,63]
        self.mid_block = ResidualBlock1D(channels[-1], channels[-1], time_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels)-1)):
            self.up_blocks.append(
                UpBlock(channels[i+1], channels[i], time_dim)
            )
        
        # 输出层 [B,32,252] -> [B,1,252]
        self.output_conv = nn.Conv1d(channels[0], output_dim, kernel_size=1)

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        date: torch.Tensor, 
        market_cap: torch.Tensor
    ) -> torch.Tensor:
        # 时间/条件嵌入
        t_emb = self.time_embed(t)  # [B,time_dim]
        c_emb = self.cond_proj(date, market_cap)  # [B,time_dim]
        combined_cond = t_emb + c_emb  # [B,time_dim]
        
        # 输入处理
        x = x.unsqueeze(1) if x.dim() == 2 else x  # [B,1,252]
        x = self.input_conv(x)
        
        # 下采样路径
        skips = []
        for block in self.down_blocks:
            x = block(x, combined_cond)
            skips.append(x)
        
        # 中间层
        x = self.mid_block(x, combined_cond)
        
        # 上采样路径
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip, combined_cond)
        
        return self.output_conv(x).squeeze(1)

# 7. 扩散过程 =========================================================
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
    
    def add_noise(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回加噪样本和噪声"""
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        noisy_x = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return noisy_x, noise

# 8. 数据加载 =========================================================
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

# 9. 训练函数 =========================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=1000, device=device)
    model = ConditionalUNet1D(channels=[32, 64, 128]).to(device)
    
    # 优化器配置 (完整配置)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100,  # 半周期长度
        eta_min=1e-6  # 最小学习率
    )
    
    # 数据加载
    dataset = FinancialDataset("financial_data/sequences/sequences_252.pt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Loaded {len(dataset)} samples")
    
    # 训练循环
    for epoch in range(1, 101):
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
        
        # 更新学习率
        scheduler.step()
        
        # 打印统计信息
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "conditional_diffusion_1d.pth")

if __name__ == "__main__":
    train()