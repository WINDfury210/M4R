import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from math import sqrt
from typing import Optional, Tuple

class TimeEmbedding(nn.Module):
    """Sinusoidal时间位置编码"""
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Time embedding dim must be even"
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ConditionEncoder(nn.Module):
    """多尺度条件特征提取器"""
    def __init__(self, input_dim=1, hidden_dims=[64, 128, 256]):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(prev_dim, dim, 3, stride=2, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.SiLU(),
                    nn.Dropout(0.1)
                )
            )
            prev_dim = dim
            
    def forward(self, x):
        # x: [B, seq_len, 1]
        features = []
        x = x.permute(0, 2, 1)  # [B, 1, seq_len]
        for block in self.conv_blocks:
            x = block(x)
            features.append(x.permute(0, 2, 1))  # 恢复[B, seq_len, C]
        return features  # 多尺度特征列表

class DownBlock(nn.Module):
    """U-Net下采样块"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        
    def forward(self, x, t, cond_emb=None):
        # x: [B, C, L]
        h = self.conv(x)
        time_emb = self.time_mlp(t)[:, :, None]  # [B, C, 1]
        h = h + time_emb
        
        if cond_emb is not None:
            cond_emb = cond_emb.permute(0, 2, 1)  # [B, C, L]
            h = h + F.interpolate(cond_emb, size=h.shape[-1], mode='linear')
            
        return h

class UpBlock(nn.Module):
    """U-Net上采样块"""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        
    def forward(self, x, skip, t, cond_emb=None):
        # x: [B, C, L]
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        h = self.conv(x)
        time_emb = self.time_mlp(t)[:, :, None]  # [B, C, 1]
        h = h + time_emb
        
        if cond_emb is not None:
            cond_emb = cond_emb.permute(0, 2, 1)  # [B, C, L]
            h = h + F.interpolate(cond_emb, size=h.shape[-1], mode='linear')
            
        return h

class TransformerBlock(nn.Module):
    """轻量级Transformer块"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x: [B, L, C]
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class FinancialDiffusion(nn.Module):
    """混合U-Net+Transformer扩散模型"""
    def __init__(self, seq_len=252, time_dim=256, cond_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.time_embed = TimeEmbedding(time_dim)
        
        # 条件编码器
        self.cond_encoder = ConditionEncoder(input_dim=1, hidden_dims=[64, 128, 256])
        self.cond_proj = nn.Linear(256, cond_dim)
        
        # 下采样路径
        self.down1 = DownBlock(1, 64, time_dim)
        self.down2 = DownBlock(64, 128, time_dim)
        self.down3 = DownBlock(128, 256, time_dim)
        
        # 瓶颈层Transformer
        self.mid_transformer = TransformerBlock(256)
        
        # 上采样路径
        self.up1 = UpBlock(256 + 128, 128, time_dim)
        self.up2 = UpBlock(128 + 64, 64, time_dim)
        self.up3 = UpBlock(64, 64, time_dim)
        
        # 输出层
        self.final_conv = nn.Conv1d(64, 1, 3, padding=1)
        self.final_norm = nn.LayerNorm(seq_len)
        
    def forward(self, x, t, cond):
        # x: [B, seq_len] - 噪声序列
        # cond: [B, seq_len] - 条件序列
        
        # 时间嵌入
        t_emb = self.time_embed(t)  # [B, time_dim]
        
        # 多尺度条件特征
        cond_features = self.cond_encoder(cond.unsqueeze(-1))  # 3个[B, L_i, C_i]
        cond_embs = [self.cond_proj(f) for f in cond_features]
        
        # 下采样
        x = x.unsqueeze(1)  # [B, 1, L]
        d1 = self.down1(x, t_emb, cond_embs[0].permute(0, 2, 1))  # [B, 64, L/2]
        d2 = self.down2(d1, t_emb, cond_embs[1].permute(0, 2, 1))  # [B, 128, L/4]
        d3 = self.down3(d2, t_emb, cond_embs[2].permute(0, 2, 1))  # [B, 256, L/8]
        
        # Transformer处理
        d3 = d3.permute(0, 2, 1)  # [B, L/8, 256]
        d3 = self.mid_transformer(d3)
        d3 = d3.permute(0, 2, 1)  # [B, 256, L/8]
        
        # 上采样
        u1 = self.up1(d3, d2, t_emb)  # [B, 128, L/4]
        u2 = self.up2(u1, d1, t_emb)  # [B, 64, L/2]
        u3 = self.up3(u2, None, t_emb)  # [B, 64, L]
        
        # 输出
        out = self.final_conv(u3).squeeze(1)  # [B, L]
        return self.final_norm(out)

class DiffusionTrainer:
    """完整的扩散训练系统"""
    def __init__(self, model, seq_len=252, num_timesteps=200, device='cuda'):
        self.model = model.to(device)
        self.seq_len = seq_len
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 定义噪声调度
        self.betas = self._cosine_schedule(num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def _cosine_schedule(self, n_steps, s=0.008):
        """余弦噪声调度"""
        steps = torch.arange(n_steps + 1, dtype=torch.float32) / n_steps
        f = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ​**​ 2
        return torch.clip((f[1:] / f[:-1]), 0.0001, 0.9999)
    
    def train_step(self, x0, cond, optimizer):
        """单次训练迭代"""
        self.model.train()
        optimizer.zero_grad()
        
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=self.device)
        
        # 前向加噪
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        
        # 预测噪声
        pred_noise = self.model(xt, t, cond)
        
        # 损失计算（含金融特性正则项）
        mse_loss = F.mse_loss(pred_noise, noise)
        
        # 金融数据特性增强
        gen_returns = xt - pred_noise
        skew_loss = torch.abs(torch.mean(gen_returns**3))  # 偏态约束
        kurt_loss = torch.abs(torch.mean(gen_returns**4) - 3)  # 超峰态约束
        
        loss = mse_loss + 0.01 * skew_loss + 0.01 * kurt_loss
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return {'loss': loss.item(), 
                'mse': mse_loss.item(),
                'skew': skew_loss.item(),
                'kurtosis': kurt_loss.item()}
    
    def sample(self, cond, steps=100, method='ddim'):
        """生成样本"""
        self.model.eval()
        with torch.no_grad():
            # 初始噪声
            x = torch.randn(cond.shape[0], self.seq_len, device=self.device)
            
            # 时间步安排
            step_indices = torch.linspace(0, self.num_timesteps-1, steps, dtype=torch.long)
            
            for i in reversed(range(steps)):
                t = torch.full((cond.shape[0],), step_indices[i], device=self.device)
                
                # 预测噪声
                pred_noise = self.model(x, t, cond)
                
                # 去噪步骤
                alpha_bar = self.alpha_bars[t].view(-1, 1)
                alpha_bar_prev = self.alpha_bars[step_indices[max(i-1, 0)]].view(-1, 1)
                
                if method == 'ddpm':
                    # DDPM标准采样
                    x = (x - (1 - alpha_bar) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
                    if i > 0:
                        noise = torch.randn_like(x)
                        sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
                        x = x + sigma * noise
                else:
                    # DDIM确定性采样
                    x = torch.sqrt(alpha_bar_prev) * (x / torch.sqrt(alpha_bar)) + \
                        torch.sqrt(1 - alpha_bar_prev - 0.1**2) * pred_noise
                        
            return x

# 示例用法
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len = 252
    
    # 初始化模型和训练器
    model = FinancialDiffusion(seq_len=seq_len).to(device)
    trainer = DiffusionTrainer(model, seq_len=seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 模拟数据加载
    def generate_batch(batch_size):
        x = torch.randn(batch_size, seq_len, device=device) * 0.02  # 真实数据应为对数收益率
        cond = torch.randn(batch_size, seq_len, device=device)  # 条件数据
        return x, cond
    
    # 训练循环
    for epoch in range(100):
        x0, cond = generate_batch(64)
        metrics = trainer.train_step(x0, cond, optimizer)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, MSE={metrics['mse']:.4f}")
            
    # 生成样本
    test_cond = torch.randn(5, seq_len, device=device)
    samples = trainer.sample(test_cond, steps=100)
    print(f"Generated samples shape: {samples.shape}")