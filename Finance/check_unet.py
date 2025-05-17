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

# ==================== 模型架构（与训练代码完全一致）====================
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        emb = time[:, None] * self.emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class FinancialDiffusion(nn.Module):
    def __init__(
        self,
        seq_len = 252,
        cond_dim = 4,
        time_dim = 256,
        channels = [32, 64, 128, 256, 512],
        num_blocks = 2,
        activation = "silu"
    ):
        super().__init__()
        self.seq_len = seq_len
        
        # 验证序列长度可被下采样次数整除
        n_down = len(channels) - 1
        assert (seq_len % (2 ** n_down)) == 0, \
            f"Sequence length {seq_len} must be divisible by {2 ** n_down} for {n_down} downsamplings"
        
        # 激活函数
        self.act = {
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "relu": nn.ReLU()
        }[activation]

        # 时间编码
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            self.act,
            nn.Linear(time_dim, channels[0])
        )
        
        # 条件编码
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            self.act,
            nn.Linear(time_dim, channels[0])
        )
        
        # 输入层
        self.input_conv = nn.Conv1d(1, channels[0], 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.down_blocks.append(DownBlock(
                in_ch=channels[i],
                out_ch=channels[i+1],
                num_blocks=num_blocks,
                act=self.act
            ))
        
        # 中间层
        self.mid_block = MidBlock(
            channels=channels[-1],
            num_blocks=num_blocks*2,
            act=self.act
        )
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channels)-1)):
            self.up_blocks.append(UpBlock(
                in_ch=channels[i+1],
                out_ch=channels[i],
                skip_ch=channels[i],
                num_blocks=num_blocks,
                act=self.act
            ))
        
        # 输出层
        self.output = nn.Conv1d(channels[0], 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 保存原始长度
        orig_len = x.shape[-1]
        
        # 时间条件
        t_emb = self.time_mlp(self.time_embed(t)).unsqueeze(-1)
        
        # 项目条件
        c_emb = self.cond_proj(cond).unsqueeze(-1)
        
        # 初始卷积
        x = self.input_conv(x.unsqueeze(1)) + t_emb + c_emb
        
        # 下采样并保存跳跃连接
        skips = []
        for block in self.down_blocks:
            skips.append(x)
            x = block(x)
        
        # 中间层
        x = self.mid_block(x)
        
        # 上采样并与跳跃连接拼接
        for i, block in enumerate(self.up_blocks):
            x = block(x, skips[-(i+1)])
        
        # 确保输出长度与输入一致
        if x.shape[-1] != orig_len:
            x = F.interpolate(x, size=orig_len, mode='linear')
        
        return self.output(x).squeeze(1)

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, act: nn.Module):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                act
            ])
            in_ch = out_ch
        layers.append(nn.MaxPool1d(2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        skip_ch: int,
        num_blocks: int, 
        act: nn.Module
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, 1)
        
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv1d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                act
            ])
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # 调整skip连接尺寸
        if skip.shape[-1] != x.shape[-1]:
            skip = F.interpolate(skip, size=x.shape[-1], mode='linear')
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.blocks(x)

class MidBlock(nn.Module):
    def __init__(self, channels: int, num_blocks: int, act: nn.Module):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                act
            ])
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ==================== 扩散过程（与训练代码完全一致）====================
class DiffusionProcess(nn.Module):
    def __init__(self, num_timesteps: int = 1000, device: str = "cpu"):
        super().__init__()
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
    
    def sample(self, model: nn.Module, cond: torch.Tensor, steps: int = 100) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            x = torch.randn(cond.shape[0], model.seq_len, device=cond.device)
            
            for t in tqdm(reversed(range(0, self.num_timesteps, self.num_timesteps//steps)), 
                         desc="Sampling"):
                t_batch = torch.full((cond.shape[0],), t, device=cond.device)
                pred_noise = model(x, t_batch, cond)
                
                alpha_bar = self.alpha_bars[t].view(-1, 1)
                alpha_bar_prev = self.alpha_bars[max(t-1, 0)].view(-1, 1)
                
                x = (x - (1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
                    x += sigma * noise
            
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
        
        # 初始化模型（必须与训练时完全相同的配置）
        model_config = {
            "seq_len": SEQUENCE_LENGTH,
            "cond_dim": 4,
            "time_dim": 256,
            "channels": [32, 64, 128, 256, 512],
            "num_blocks": 2,
            "activation": "silu"
        }
        model = FinancialDiffusion(**model_config).to(DEVICE)
        
        # 加载模型权重
        logger.info("Loading model weights...")
        state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully!")
        
        # 初始化扩散过程
        diffusion = DiffusionProcess(num_timesteps=1000, device=DEVICE)
        
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