import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# 定义时间嵌入模块
class TimeEmbedding(nn.Module):
    def __init__(self, dim, embedding_type="linear", use_time_embedding=True, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.embedding_type = embedding_type
        self.use_time_embedding = use_time_embedding

        if embedding_type == "sinusoidal":
            half_dim = dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            self.register_buffer("emb", emb)
        elif embedding_type == "linear":
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )

    def forward(self, time):
        if not self.use_time_embedding:
            return torch.zeros(time.size(0), self.dim, device=time.device)

        if self.embedding_type == "sinusoidal":
            emb = time[:, None] * self.emb[None, :]
            emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1, 0, 0))
            return emb
        elif self.embedding_type == "linear":
            time = time.unsqueeze(-1).float()
            return self.mlp(time)

# 定义条件 UNet 模型（匹配训练模型）
class ConditionalUNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, time_dim=128, channels=[32, 64, 64, 128, 128, 256], use_time_embedding=True, time_embedding_type="linear"):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, embedding_type=time_embedding_type, use_time_embedding=use_time_embedding)
        self.label_embedding = nn.Embedding(num_classes, time_dim)

        self.channels = channels
        self.num_layers = len(channels)

        # 编码器
        self.encoder_convs = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        in_channels = input_dim
        for out_channels in channels:
            self.encoder_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.encoder_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # 解码器（通道数匹配训练模型）
        decoder_channels = [384, 256, 192, 128, 96, 64]
        self.decoder_convs = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        in_channels = channels[-1]
        for out_channels in decoder_channels[:-1]:
            self.decoder_convs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.decoder_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self.final_conv = nn.ConvTranspose2d(in_channels, output_dim, kernel_size=3, padding=1)

        # 标签和时间嵌入投影
        self.fc_time = nn.Linear(time_dim, channels[-1])
        self.fc_label = nn.Linear(time_dim, channels[-1])

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.final_conv.weight, mean=0, std=0.01)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, time, labels):
        time_emb = self.time_embedding(time)
        label_emb = self.label_embedding(labels)

        time_emb = self.fc_time(time_emb).view(-1, self.channels[-1], 1, 1)
        label_emb = self.fc_label(label_emb).view(-1, self.channels[-1], 1, 1)

        # 编码器
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            x = F.relu(norm(conv(x)))

        # 添加条件信息
        x = x + time_emb + label_emb

        # 解码器
        for conv, norm in zip(self.decoder_convs, self.decoder_norms):
            x = F.relu(norm(conv(x)))
        x = self.final_conv(x) * 0.1  # 缩放输出

        return x

# 定义扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, time_dim=128, channels=[32, 64, 64, 128, 128, 256], use_time_embedding=True, time_embedding_type="linear"):
        super().__init__()
        self.unet = ConditionalUNet(input_dim, output_dim, num_classes, time_dim, channels, use_time_embedding, time_embedding_type)

    def forward(self, x, time, labels):
        return self.unet(x, time, labels)

# 定义扩散过程（与训练一致）
class DiffusionProcess:
    def __init__(self, num_timesteps=500, beta_min=0.0001, beta_max=0.01, schedule="linear", device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        t = torch.linspace(0, 1, num_timesteps + 1, device=device, dtype=torch.float32)[:-1]
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_min, beta_max, num_timesteps, device=device)
            integrals = torch.cumsum(self.betas * (1.0 / num_timesteps), dim=0)
            self.alphas = torch.exp(-0.5 * integrals)
            self.sigma_squares = 1 - torch.exp(-integrals)
        self.sqrt_sigmas = torch.sqrt(self.sigma_squares)

    def add_noise(self, x, t, clamp_range=(-1, 1)):
        alpha_t = self.alphas[t].view(-1, *([1] * (x.dim() - 1)))
        sigma_t = self.sqrt_sigmas[t].view(-1, *([1] * (x.dim() - 1)))
        noise = torch.randn_like(x, device=self.device)
        noisy_x = torch.sqrt(alpha_t) * x + sigma_t * noise
        if clamp_range:
            noisy_x = torch.clamp(noisy_x, *clamp_range)
        return noisy_x, noise

# 生成函数
@torch.no_grad()
def generate(model, diffusion, labels, device, input_shape, steps=100, method="ddpm", eta=0.0, lambda_corrector=0.01, clamp_range=(-1, 1)):
    model.eval()
    x = torch.randn((labels.size(0), *input_shape), device=device, dtype=torch.float32)
    step_indices = torch.linspace(diffusion.num_timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)
    
    for i in tqdm(range(steps), desc=f"Generating Steps ({method})"):
        t = step_indices[i]
        t_next = step_indices[i + 1]
        t_tensor = torch.full((labels.size(0),), t, device=device)
        pred_noise = model(x, t_tensor, labels)
        
        if i % 20 == 0:
            print(f"Step {i}: pred_noise mean={pred_noise.mean().item():.4f}, std={pred_noise.std().item():.4f}")
        
        alpha_t = diffusion.alphas[t].view(-1, *([1] * (x.dim() - 1)))
        sigma_t = diffusion.sqrt_sigmas[t].view(-1, *([1] * (x.dim() - 1)))
        alpha_t_next = diffusion.alphas[t_next].view(-1, *([1] * (x.dim() - 1)))
        sigma_t_next = diffusion.sqrt_sigmas[t_next].view(-1, *([1] * (x.dim() - 1)))
        
        s_theta = -pred_noise / sigma_t
        f_t = -0.5 * diffusion.betas[t] * x
        g_t = torch.sqrt(diffusion.betas[t])
        dt = torch.tensor(-1.0 / diffusion.num_timesteps, device=device)
        
        if method == "ddim":
            x_0_pred = (x - sigma_t * pred_noise) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_t_next) * x_0_pred + sigma_t_next * pred_noise
        elif method == "hybrid":
            x_0_pred = (x - sigma_t * pred_noise) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_t_next) * x_0_pred + sigma_t_next * pred_noise + eta * sigma_t_next * torch.randn_like(x)
        elif method == "ddpm":
            bar_alpha_t = torch.prod(diffusion.alphas[:t + 1]).view(-1, *([1] * (x.dim() - 1)))
            x = (x - (1 - alpha_t) / torch.sqrt(1 - bar_alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            if t > 0:
                x = x + sigma_t * torch.randn_like(x)
        elif method == "pc":
            x = x + (f_t - g_t**2 * s_theta) * dt + g_t * torch.sqrt(torch.abs(dt)) * torch.randn_like(x)
            if lambda_corrector > 0 and t_next > 0:
                t_next_tensor = torch.full((labels.size(0),), t_next, device=device)
                pred_noise_next = model(x, t_next_tensor, labels)
                s_theta_next = -pred_noise_next / sigma_t_next
                x = x + lambda_corrector * s_theta_next + torch.sqrt(torch.tensor(2 * lambda_corrector, device=device)) * torch.randn_like(x)
        
        if clamp_range:
            x = torch.clamp(x, *clamp_range)
        
        if i % 20 == 0:
            tqdm.write(f"Step {i}: min={x.min().item():.4f}, max={x.max().item():.4f}")
    
    return x

# 主程序（仅生成部分，逐方法保存）
if __name__ == "__main__":
    # 初始化设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionModel(
        input_dim=1,
        output_dim=1,
        num_classes=10,
        time_dim=128,
        channels=[32, 64, 64, 128, 128, 256],
        use_time_embedding=True,
        time_embedding_type="linear"
    ).to(device)
    
    # 加载训练好的模型
    model.load_state_dict(torch.load("models/cond_mnist_vp.pth", map_location=device))
    model.eval()
    
    # 初始化扩散过程
    diffusion = DiffusionProcess(num_timesteps=500, beta_min=0.0001, beta_max=0.01, schedule="linear", device=device)
    
    # 设置生成参数
    num_samples_per_digit = 10
    digits = list(range(10))
    labels = torch.tensor([i for i in digits for _ in range(num_samples_per_digit)], dtype=torch.long, device=device)
    input_shape = (1, 28, 28)
    sampling_methods = [
        {"method": "ddim", "eta": 0.0, "lambda_corrector": 0.0},
        {"method": "ddpm", "eta": 0.0, "lambda_corrector": 0.0},
        {"method": "hybrid", "eta": 0.5, "lambda_corrector": 0.0},
        {"method": "pc", "eta": 0.0, "lambda_corrector": 0.01}
    ]
    
    # 生成并逐方法保存样本
    with torch.no_grad():
        for method_config in sampling_methods:
            method = method_config["method"]
            images = generate(
                model, diffusion, labels, device, input_shape,
                steps=100, method=method, eta=method_config["eta"],
                lambda_corrector=method_config["lambda_corrector"],
                clamp_range=(-1, 1)
            )
            images_np = images.cpu().numpy()
            print(f"Generated {method} images: min={images.min().item():.4f}, max={images.max().item():.4f}")
            
            # 立即保存图像
            images_display = (images_np + 1) / 2
            images_display = np.clip(images_display, 0, 1)
            
            fig, axes = plt.subplots(len(digits), num_samples_per_digit, figsize=(num_samples_per_digit * 2, len(digits) * 2))
            for i, digit in enumerate(digits):
                for j in range(num_samples_per_digit):
                    idx = i * num_samples_per_digit + j
                    axes[i, j].imshow(images_display[idx, 0], cmap="gray", vmin=0, vmax=1)
                    axes[i, j].set_title(f"Sample {j+1}")
                    if j == 0:
                        axes[i, j].set_ylabel(f"Digit {digit}")
                    axes[i, j].axis("off")
            plt.suptitle(f"Sampling Method: {method.upper()}")
            plt.tight_layout()
            
            os.makedirs("images", exist_ok=True)
            plt.savefig(f"images/generated_mnist_{method}.png")
            print(f"Samples saved to images/generated_mnist_{method}.png")
            plt.close(fig)