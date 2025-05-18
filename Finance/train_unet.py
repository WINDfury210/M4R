import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        emb = time.float().view(-1, 1) * self.emb.view(1, -1)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# 3. 条件UNet模型 ======================================================
class ConditionalUNet(nn.Module):
    def __init__(self, seq_len: int = 252):
        super().__init__()
        self.seq_len = seq_len
        
        # 时间/条件嵌入
        self.time_embed = TimeEmbedding(128)
        self.cond_proj = nn.Sequential(
            nn.Linear(4, 128),  # date(3) + market_cap(1)
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # 输入层 [B,1,252]
        self.input_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # [B,64,126]
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # [B,128,63]
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        # 中间层 [B,128,63]
        self.mid_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU()
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B,64,126]
            nn.GroupNorm(8, 64),
            nn.SiLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B,32,252]
            nn.GroupNorm(8, 32),
            nn.SiLU()
        )
        
        # 输出层 [B,32,252] -> [B,1,252]
        self.output_conv = nn.Conv1d(32, 1, kernel_size=1)
        
        # 条件注入层（每层独立）
        self.cond_inject_input = nn.Linear(128, 32)
        self.cond_inject_down1 = nn.Linear(128, 64)
        self.cond_inject_down2 = nn.Linear(128, 128)
        self.cond_inject_mid = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor, t: torch.Tensor, date: torch.Tensor, market_cap: torch.Tensor) -> torch.Tensor:
        # 条件嵌入
        t_emb = self.time_embed(t)  # [B,128]
        cond = torch.cat([date, market_cap], dim=-1)  # [B,4]
        c_emb = self.cond_proj(cond)  # [B,128]
        combined_cond = t_emb + c_emb  # [B,128]
        
        # 输入处理 [B,1,252]
        x = x.unsqueeze(1)  # 确保输入是3D张量
        h0 = self.input_conv(x)
        
        # 下采样路径 + 条件注入
        h0 = h0 + self.cond_inject_input(combined_cond).unsqueeze(-1)
        h1 = self.down1(h0)
        h1 = h1 + self.cond_inject_down1(combined_cond).unsqueeze(-1)
        h2 = self.down2(h1)
        h2 = h2 + self.cond_inject_down2(combined_cond).unsqueeze(-1)
        
        # 中间层
        h_mid = self.mid_conv(h2)
        h_mid = h_mid + self.cond_inject_mid(combined_cond).unsqueeze(-1)
        
        # 上采样路径 + 跳连
        h_up1 = self.up1(h_mid)
        h_up1 = h_up1 + h1  # 跳连
        h_up2 = self.up2(h_up1)
        h_up2 = h_up2 + h0  # 跳连
        
        # 输出处理
        out = self.output_conv(h_up2)
        return out.squeeze(1)  # [B,252]

# 4. 数据加载 ==========================================================
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

# 5. 训练函数 ==========================================================
def train(num_epochs, num_timesteps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化组件
    diffusion = DiffusionProcess(num_timesteps=num_timesteps, device=device)
    model = ConditionalUNet(seq_len=252).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # 数据加载
    dataset = FinancialDataset("financial_data/sequences/sequences_252.pt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # 模型结构验证
    test_input = torch.randn(2, 252, device=device)
    test_t = torch.randint(0, 1000, (2,), device=device)
    test_date = torch.randn(2, 3, device=device)
    test_mcap = torch.randn(2, 1, device=device)
    test_output = model(test_input, test_t, test_date, test_mcap)
    assert test_output.shape == test_input.shape, \
        f"Shape mismatch: input {test_input.shape}, output {test_output.shape}"
    print("Model structure validated!")
    
    # 训练循环
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
        
        # 打印统计信息
        avg_loss = epoch_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "conditional_diffusion_final.pth")
    print("Training completed and model saved")

if __name__ == "__main__":
    train(1000, 2000)