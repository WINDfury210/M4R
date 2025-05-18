import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 扩散过程实现 ======================================================
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
        """添加噪声到输入数据"""
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        noisy = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return noisy, noise
    
    def sample_timesteps(self, n):
        """生成随机时间步"""
        return torch.randint(0, self.num_timesteps, (n,), device=self.device)

# 2. 条件UNet模型 =====================================================
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
    def __init__(self, channels=[32, 64, 128], time_dim=256):
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
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels[i], channels[i+1], 3, stride=2, padding=1),
                nn.GroupNorm(8, channels[i+1]),
                nn.SiLU()
            ) for i in range(len(channels)-1)
        ])
        
        # 中间层
        self.mid_conv = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], 3, padding=1),
            nn.GroupNorm(8, channels[-1]),
            nn.SiLU()
        )
        
        # 解码器 (带跳连)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    channels[i+1]*2, channels[i], 3, 
                    stride=2, padding=1, output_padding=1
                ),
                nn.GroupNorm(8, channels[i]),
                nn.SiLU()
            ) for i in reversed(range(len(channels)-1))
        ])
        
        # 输出层
        self.output_conv = nn.Conv1d(channels[0], 1, 1)

    def forward(self, x, t, date, market_cap):
        # 条件嵌入
        t_emb = self.time_embed(t.float())
        cond = torch.cat([date, market_cap], dim=-1)
        c_emb = self.cond_proj(cond)
        
        # 输入处理
        x = x.unsqueeze(1)
        h = [self.input_conv(x) + self.time_inject(t_emb + c_emb).unsqueeze(-1)]
        
        # 编码路径
        for block in self.encoder:
            h.append(block(h[-1]))
        
        # 中间层
        h[-1] = self.mid_conv(h[-1])
        
        # 解码路径
        for i, block in enumerate(self.decoder):
            skip = h[-(i+2)]
            h[-1] = torch.cat([h[-1], skip], dim=1)
            h.append(block(h[-1]))
        
        return self.output_conv(h[-1]).squeeze(1)

# 3. 训练流程 ==========================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=1000, device=device)
    model = ConditionalUNet(channels=[32, 64, 128, 256]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    
    # 数据加载
    dataset = FinancialDataset("financial_data/sequences_252.pt")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 训练循环
    for epoch in range(1, 501):
        for batch in dataloader:
            x = batch["sequence"].to(device)
            date = batch["date"].to(device)
            market_cap = batch["market_cap"].to(device)
            
            # 扩散过程
            t = diffusion.sample_timesteps(x.shape[0])
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

# 4. 数据加载器 ========================================================
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

if __name__ == "__main__":
    train()