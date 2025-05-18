import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 1. 扩散过程 ==========================================================
class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 线性噪声调度
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

# 2. 时间嵌入 ==========================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
    def forward(self, time):
        emb = time.float().view(-1, 1) * self.emb.view(1, -1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# 3. 残差块 ============================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
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
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
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
    
    def forward(self, x, cond):
        h = self.block1(x)
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.block2(h)
        return h + self.residual(x)

# 4. 下采样/上采样层 ==================================================
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        return self.conv(x)

# 5. 可配置的Conditional UNet =========================================
class ConditionalUNet(nn.Module):
    def __init__(self, 
                 base_channels=32,
                 channel_mults=(1, 2, 4),
                 num_res_blocks=2,
                 cond_dim=128):
        super().__init__()
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
        skip_channels = [base_channels]
        
        # 构建下采样路径
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(skip_channels[-1], out_channels, cond_dim))
                skip_channels.append(out_channels)
            
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(out_channels))
                skip_channels.append(out_channels)
        
        # 中间层
        self.mid_block1 = ResBlock(skip_channels[-1], skip_channels[-1], cond_dim)
        self.mid_block2 = ResBlock(skip_channels[-1], skip_channels[-1], cond_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        
        # 构建上采样路径
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                # 确保有足够的skip_channels可用
                if len(skip_channels) < 2:
                    raise ValueError("Not enough skip channels for upsampling path")
                
                in_channels = skip_channels.pop() + skip_channels[-1]
                self.up_blocks.append(ResBlock(in_channels, out_channels, cond_dim))
            
            if i != 0:
                self.up_blocks.append(Upsample(out_channels))
        
        # 输出层
        self.output_conv = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def forward(self, x, t, date, market_cap):
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
        h = self.mid_block2(h, combined_cond)
        
        # 上采样
        for block in self.up_blocks:
            if isinstance(block, ResBlock):
                # 确保有足够的skip connections
                if len(skip_connections) < 2:
                    raise RuntimeError("Not enough skip connections for upsampling")
                h = block(torch.cat([h, skip_connections.pop()], dim=1), combined_cond)
            else:
                h = block(h)
        
        # 输出
        return self.output_conv(h).squeeze(1)

# 6. 数据加载 ==========================================================
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

# 7. 训练函数 ==========================================================
def train(
    num_epochs=500,
    num_timesteps=1000,
    batch_size=32,
    base_channels=32,
    channel_mults=(1, 2, 4),
    num_res_blocks=2,
    lr=2e-4,
    data_path="financial_data/sequences/sequences_252.pt",
    save_path="conditional_diffusion_final.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=num_timesteps, device=device)
    model = ConditionalUNet(
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 数据加载
    dataset = FinancialDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # 训练循环
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
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
        
        # 打印统计信息
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'base_channels': base_channels,
            'channel_mults': channel_mults,
            'num_res_blocks': num_res_blocks
        }
    }, save_path)
    print(f"Training completed. Model saved to {save_path}")

if __name__ == "__main__":
    train(
        num_epochs=1000,
        num_timesteps=2000,
        base_channels=32,
        channel_mults=(1, 2, 4, 4),  # 改为3层UNet更稳定
        num_res_blocks=2,
        lr=2e-4
    )