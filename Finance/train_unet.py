import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

# 2. 条件UNet模型 ======================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, time):
        emb = time * self.emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ConditionalUNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256], time_dim=256):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        
        # 条件投影 (日期3维 + 市值1维)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, time_dim)
        )
        
        # 输入层
        self.input_conv = nn.Conv1d(1, channels[0], 3, padding=1)
        self.time_inject = nn.Linear(time_dim, channels[0])
        
        # 编码器
        self.encoder = nn.ModuleList()
        for i in range(len(channels)-1):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(channels[i], channels[i+1], 3, stride=2, padding=1),
                    nn.GroupNorm(8, channels[i+1]),
                    nn.SiLU()
                )
            )
        
        # 中间层
        self.mid_conv = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], 3, padding=1),
            nn.GroupNorm(8, channels[-1]),
            nn.SiLU()
        )
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(channels)-1)):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        channels[i+1], channels[i], 3, 
                        stride=2, padding=1, output_padding=1
                    ),
                    nn.GroupNorm(8, channels[i]),
                    nn.SiLU()
                )
            )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], 3, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], 1, 1)
        )

    def forward(self, x, t, date, market_cap):
        # 条件嵌入
        t_emb = self.time_embed(t.float())
        cond = torch.cat([date, market_cap], dim=-1)
        c_emb = self.cond_proj(cond)
        
        # 输入处理
        x = x.unsqueeze(1)
        h = self.input_conv(x) + self.time_inject(t_emb + c_emb).unsqueeze(-1)
        
        # 编码路径
        skips = []
        for block in self.encoder:
            h = block(h)
            skips.append(h)
        
        # 中间层
        h = self.mid_conv(h)
        
        # 解码路径 (带跳连)
        for block in self.decoder:
            h = block(h)
            if skips:  # 添加跳连
                skip = skips.pop()
                # 确保尺寸匹配
                if h.shape[-1] != skip.shape[-1]:
                    h = nn.functional.interpolate(h, size=skip.shape[-1], mode='linear')
                h = h + skip
        
        return self.output_conv(h).squeeze(1)

# 3. 数据加载 ==========================================================
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

# 4. 训练函数 ==========================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=1000, device=device)
    model = ConditionalUNet(channels=[32, 64, 128, 256, 512]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    
    # 数据加载 - 使用您提供的正确路径
    try:
        dataset = FinancialDataset("financial_data/sequences/sequences_252.pt")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"Loaded dataset with {len(dataset)} samples")
        print(f"Sample shape: {dataset[0]['sequence'].shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 训练循环
    for epoch in range(1, 501):
        for batch_idx, batch in enumerate(dataloader):
            x = batch["sequence"].to(device)
            date = batch["date"].to(device)
            market_cap = batch["market_cap"].to(device)
            
            # 扩散过程
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            noisy_x, noise = diffusion.add_noise(x, t)
            
            # 训练步骤
            pred_noise = model(noisy_x, t, date, market_cap)
            loss = nn.functional.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
    
    torch.save(model.state_dict(), "conditional_diffusion.pth")
    print("Training completed and model saved")

if __name__ == "__main__":
    train()