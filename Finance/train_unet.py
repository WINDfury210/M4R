import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 模型架构 ====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, time):
        emb = time[:, None] * self.emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class FinancialUNet(nn.Module):
    def __init__(self, seq_len=252, cond_dim=4, time_dim=128):
        super().__init__()
        self.seq_len = seq_len
        
        # 时间编码
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 条件编码 (3个日期特征 + 1个市值特征)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(time_dim, time_dim)
        )
        
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        # 中间层
        self.mid = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Conv1d(128+64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.up2 = nn.Sequential(
            nn.Conv1d(64+32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        
        # 输出层
        self.output = nn.Conv1d(32, 1, 1)
        
    def forward(self, x, t, cond):
        x = x.unsqueeze(1)  # [B, 1, seq_len]
        
        # 时间条件
        t_emb = self.time_embed(t).unsqueeze(-1)  # [B, time_dim, 1]
        
        # 项目条件
        c_emb = self.cond_proj(cond).unsqueeze(-1)  # [B, time_dim, 1]
        
        # 下采样
        x1 = self.down1(x) + t_emb + c_emb
        x2 = self.down2(F.max_pool1d(x1, 2)) + t_emb + c_emb
        
        # 中间层
        x_mid = self.mid(F.max_pool1d(x2, 2)) + t_emb + c_emb
        
        # 上采样
        x_up = F.interpolate(x_mid, scale_factor=2, mode='linear')
        x_up = self.up1(torch.cat([x_up, x2], dim=1)) + t_emb + c_emb
        
        x_up = F.interpolate(x_up, scale_factor=2, mode='linear')
        x_up = self.up2(torch.cat([x_up, x1], dim=1)) + t_emb + c_emb
        
        return self.output(x_up).squeeze(1)

# ==================== 扩散过程 ====================
class DiffusionProcess(nn.Module):  # 修改为继承自nn.Module
    def __init__(self, num_timesteps=200, device="cpu"):
        super().__init__()  # 调用父类初始化
        
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
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        noisy = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return noisy, noise

# ==================== 训练函数 ====================
def train_and_save(data_path, save_dir="saved_models", epochs=100, batch_size=64):
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    data = torch.load(data_path)
    sequences = data["sequences"].float()
    start_dates = data["start_dates"].float()
    market_caps = data["market_caps"].float()
    
    # 构建条件向量 [日期特征(3) + 市值(1)]
    conditions = torch.cat([start_dates, market_caps], dim=1)
    
    # 数据集
    dataset = torch.utils.data.TensorDataset(sequences, conditions)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = FinancialUNet(seq_len=252, cond_dim=4).to(device)
    diffusion = DiffusionProcess(num_timesteps=200, device=device).to(device)  # 确保扩散模型也在正确设备上
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, cond in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, cond = x.to(device), cond.to(device)
            
            # 随机时间步
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            
            # 添加噪声
            noisy_x, noise = diffusion.add_noise(x, t)
            
            # 预测噪声
            pred_noise = model(noisy_x, t, cond)
            
            # 损失计算
            loss = F.mse_loss(pred_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 每20个epoch保存一次
        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(save_dir, f"financial_unet_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
    
    logger.info("Training completed")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "financial_data/sequences/sequences_252.pt"
    SAVE_DIR = "saved_models"
    EPOCHS = 100
    BATCH_SIZE = 64
    
    # 开始训练
    try:
        train_and_save(
            data_path=DATA_PATH,
            save_dir=SAVE_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise