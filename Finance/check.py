import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.stats import ks_2samp, skew, kurtosis, wasserstein_distance
import os
import json
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
SEQUENCE_LENGTH = 252
NUM_SAMPLES = 1000
STEPS = 200
DATA_FILE = "financial_data/sequences/sequences_252.pt"
MODEL_FILE = "saved_models/best_model.pth"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 模型架构（必须与训练代码完全一致）====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
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
        self.time_dim = time_dim
        
        # 时间编码（保持与训练代码完全一致）
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 32),
            nn.LeakyReLU(0.2)
        )
        
        # 条件编码（保持与训练代码完全一致）
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(time_dim, time_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim, 32),
            nn.LeakyReLU(0.2)
        )
        
        # 网络结构（保持与训练代码完全一致）
        self.init_conv = nn.Conv1d(1, 32, 3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.mid = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
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
        
        self.output = nn.Conv1d(32, 1, 1)

    def forward(self, x, t, cond):
        x = x.unsqueeze(1)
        
        # 处理条件（保持与训练代码完全一致）
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(-1)
        c_emb = self.cond_mlp(self.cond_proj(cond)).unsqueeze(-1)
        
        x = self.init_conv(x) + t_emb + c_emb
        
        x1 = self.down1(F.max_pool1d(x, 2))
        x2 = self.down2(F.max_pool1d(x1, 2))
        
        x_mid = self.mid(x2)
        
        x_up = self.up1(torch.cat([
            F.interpolate(x_mid, scale_factor=2, mode='linear'), 
            x1
        ], dim=1))
        
        x_out = self.up2(torch.cat([
            F.interpolate(x_up, scale_factor=2, mode='linear'),
            x
        ], dim=1))
        
        return self.output(x_out).squeeze(1)

# ==================== 扩散过程（保持与训练代码完全一致）====================
class DiffusionProcess(nn.Module):
    def __init__(self, num_timesteps=200, device="cpu"):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 余弦调度（保持与训练代码完全一致）
        t = torch.linspace(0, 1, num_timesteps+1)
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        betas = torch.clip(1 - (f[1:] / f[:-1]), 0, 0.999)
        
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1. - alpha_bars)
        
        # 注册缓冲区（保持与训练代码完全一致）
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alpha_bars', alpha_bars.to(device))
        self.register_buffer('sqrt_alpha_bars', sqrt_alpha_bars.to(device))
        self.register_buffer('sqrt_one_minus_alpha_bars', sqrt_one_minus_alpha_bars.to(device))
    
    def sample(self, model, cond, steps=100, eta=0.0):
        model.eval()
        with torch.no_grad():
            x = torch.randn(cond.shape[0], SEQUENCE_LENGTH, device=cond.device)
            
            for t in tqdm(reversed(range(0, self.num_timesteps, self.num_timesteps//steps)), 
                         desc="Sampling"):
                t_batch = torch.full((cond.shape[0],), t, device=cond.device)
                pred_noise = model(x, t_batch, cond)
                
                alpha_bar = self.alpha_bars[t].view(-1, 1)
                alpha_bar_prev = self.alpha_bars[max(t-1, 0)].view(-1, 1)
                
                x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
                
                sigma = eta * torch.sqrt((1 - alpha_bar_prev)/(1 - alpha_bar)) * torch.sqrt(1 - alpha_bar/alpha_bar_prev)
                
                x = torch.sqrt(alpha_bar_prev) * x0_pred
                if t > 0:
                    x += torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise
                    x += sigma * torch.randn_like(x)
            
            return x

# ==================== 数据加载 ====================
def load_data():
    data = torch.load(DATA_FILE)
    sequences = data["sequences"].float()
    start_dates = data["start_dates"].float()
    market_caps = data["market_caps"].float()
    conditions = torch.cat([start_dates, market_caps], dim=1)
    return sequences, conditions

# ==================== 评估指标 ====================
def calculate_metrics(real_data, generated_data):
    real_np = real_data.cpu().numpy()
    gen_np = generated_data.cpu().numpy()
    
    metrics = {
        'basic_stats': {
            'real_mean': float(real_np.mean()),
            'gen_mean': float(gen_np.mean()),
            'real_std': float(real_np.std()),
            'gen_std': float(gen_np.std()),
            'real_skew': float(skew(real_np.flatten())),
            'gen_skew': float(skew(gen_np.flatten())),
            'real_kurtosis': float(kurtosis(real_np.flatten())),
            'gen_kurtosis': float(kurtosis(gen_np.flatten()))
        },
        'distribution': {
            'ks_stat': float(ks_2samp(real_np.flatten(), gen_np.flatten())[0]),
            'wasserstein': float(wasserstein_distance(real_np.flatten(), gen_np.flatten()))
        },
        'temporal': {
            'acf_real': acf(real_np[0], nlags=20).tolist(),
            'acf_gen': acf(gen_np[0], nlags=20).tolist(),
            'acf_mse': float(np.mean((acf(real_np[0], nlags=20) - acf(gen_np[0], nlags=20)) ** 2)),
            'volatility_corr': float(np.corrcoef(
                [np.std(real_np[:, i:i+20], axis=1).mean() for i in range(real_np.shape[1]-20)],
                [np.std(gen_np[:, i:i+20], axis=1).mean() for i in range(gen_np.shape[1]-20)]
            )[0,1])
        }
    }
    return metrics

# ==================== 可视化 ====================
def visualize_results(real_data, generated_data, metrics, output_dir):
    # 样本路径
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.plot(generated_data[i], alpha=0.6, label='Generated' if i==0 else "")
        plt.plot(real_data[i], alpha=0.6, label='Real' if i==0 else "")
    plt.title("Sample Paths Comparison")
    plt.xlabel("Time")
    plt.ylabel("Returns")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "sample_paths.png"))
    plt.close()
    
    # 自相关函数
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['temporal']['acf_real'], label='Real')
    plt.plot(metrics['temporal']['acf_gen'], label='Generated')
    plt.title("Autocorrelation Function")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "acf_comparison.png"))
    plt.close()

# ==================== 主函数 ====================
def main():
    try:
        # 加载数据
        logger.info("Loading data...")
        sequences, conditions = load_data()
        
        # 加载模型（严格匹配训练时的结构）
        logger.info("Loading model...")
        model = FinancialUNet(seq_len=SEQUENCE_LENGTH, cond_dim=4).to(DEVICE)
        
        # 检查模型参数是否匹配
        state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
        
        # 调试：打印模型和state_dict的键
        logger.info("\nModel state dict keys:")
        logger.info("\n".join(model.state_dict().keys()))
        logger.info("\nLoaded state dict keys:")
        logger.info("\n".join(state_dict.keys()))
        
        # 加载权重
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model loaded successfully!")
        
        # 初始化扩散过程
        diffusion = DiffusionProcess(num_timesteps=200, device=DEVICE)
        
        # 生成样本
        logger.info(f"Generating {NUM_SAMPLES} samples...")
        with torch.no_grad():
            generated_samples = diffusion.sample(
                model, 
                conditions[:NUM_SAMPLES].to(DEVICE),
                steps=STEPS
            )
        
        # 计算评估指标
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(sequences[:NUM_SAMPLES], generated_samples)
        
        # 保存结果
        logger.info("Saving results...")
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        torch.save(generated_samples, os.path.join(OUTPUT_DIR, "generated_samples.pt"))
        visualize_results(sequences[:5].cpu(), generated_samples[:5].cpu(), metrics, OUTPUT_DIR)
        
        # 打印摘要
        logger.info("\nEvaluation Summary:")
        logger.info(f"Mean: Real={metrics['basic_stats']['real_mean']:.4f}, Generated={metrics['basic_stats']['gen_mean']:.4f}")
        logger.info(f"Std: Real={metrics['basic_stats']['real_std']:.4f}, Generated={metrics['basic_stats']['gen_std']:.4f}")
        logger.info(f"KS Statistic: {metrics['distribution']['ks_stat']:.4f}")
        logger.info(f"Wasserstein Distance: {metrics['distribution']['wasserstein']:.4f}")
        logger.info(f"ACF MSE: {metrics['temporal']['acf_mse']:.4f}")
        logger.info(f"Volatility Correlation: {metrics['temporal']['volatility_corr']:.4f}")
        
        logger.info(f"\nResults saved to {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()