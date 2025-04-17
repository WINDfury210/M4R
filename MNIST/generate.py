import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

class TimeEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        # MLP 结构：输入 1 维时间，输出 dim 维嵌入
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 输入时间 t（标量）
            nn.ReLU(),                 # 非线性激活
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)  # 输出到目标维度
        )

    def forward(self, time):
        # time: [batch_size] -> [batch_size, 1]
        time = time.unsqueeze(-1).float()
        emb = self.mlp(time)  # [batch_size, dim]
        return emb

# 定义调整后的条件 UNet 模型（适配 28x28）
class ConditionalUNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, num_styles_per_class=5, time_dim=128, channels=[64, 128, 256, 512]):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.label_embedding = nn.Embedding(num_classes, time_dim)
        self.style_embeddings = nn.ModuleList([nn.Embedding(num_styles_per_class, time_dim) for _ in range(num_classes)])

        self.channels = channels
        self.num_layers = len(channels)

        # 编码器（适配 28x28）
        self.encoder_convs = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        in_channels = input_dim
        for out_channels in channels:
            self.encoder_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.encoder_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # 解码器（适配 28x28）
        self.decoder_convs = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_channels = channels[-1 - i] + channels[-2 - i]
            out_channels = channels[-2 - i]
            self.decoder_convs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.decoder_norms.append(nn.BatchNorm2d(out_channels))
        self.final_conv = nn.ConvTranspose2d(channels[0] + channels[0], output_dim, kernel_size=3, padding=1)

        self.fc_time = nn.Linear(time_dim, channels[-1])
        self.fc_label = nn.Linear(time_dim, channels[-1])
        self.fc_style = nn.Linear(time_dim, channels[-1])

    def forward(self, x, time, labels, styles):
        time_emb = self.time_embedding(time)
        label_emb = self.label_embedding(labels)
        style_emb = torch.zeros(x.size(0), self.fc_style.in_features, device=x.device, dtype=torch.float32)
        for cls in range(len(self.style_embeddings)):
            mask = (labels == cls)
            if mask.any():
                style_emb[mask] = self.style_embeddings[cls](styles[mask])

        time_emb = self.fc_time(time_emb).view(-1, self.channels[-1], 1, 1)
        label_emb = self.fc_label(label_emb).view(-1, self.channels[-1], 1, 1)
        style_emb = self.fc_style(style_emb).view(-1, self.channels[-1], 1, 1)

        skips = []
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            residual = x
            x = F.relu(norm(conv(x)))
            if residual.shape[1] == x.shape[1]:
                x = x + residual
            skips.append(x)

        x = x + time_emb + label_emb + style_emb

        for i, (conv, norm) in enumerate(zip(self.decoder_convs, self.decoder_norms)):
            skip = skips[-2 - i]
            x = torch.cat([x, skip], dim=1)
            residual = x
            x = F.relu(norm(conv(x)))
            if residual.shape[1] == x.shape[1]:
                x = x + residual
        x = torch.cat([x, skips[0]], dim=1)
        x = self.final_conv(x)

        return x

# 定义扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, num_styles_per_class=5, time_dim=128, channels=[64, 128, 256, 512]):
        super().__init__()
        self.unet = ConditionalUNet(input_dim, output_dim, num_classes, num_styles_per_class, time_dim, channels)

    def forward(self, x, time, labels, styles):
        return self.unet(x, time, labels, styles)

# 定义余弦调度的扩散过程
class DiffusionProcess:
    def __init__(self, num_timesteps=500, device="cpu"):
        self.num_timesteps = num_timesteps
        t = torch.linspace(0, 1, num_timesteps, device=device, dtype=torch.float32)
        betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device, dtype=torch.float32)
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        noisy_x = torch.clamp(noisy_x, -1, 1)
        return noisy_x, noise

# DDIM 采样函数（适配 28x28）
@torch.no_grad()
def generate(model, diffusion, labels, styles, device, image_size=(1, 28, 28), steps=100):
    model.eval()
    x = torch.randn((labels.size(0), *image_size), device=device, dtype=torch.float32)
    timesteps = torch.linspace(diffusion.num_timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)

    for i in tqdm(range(steps), desc="Generating Steps"):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        t_tensor = torch.full((labels.size(0),), t, device=device)
        pred_noise = model(x, t_tensor, labels, styles)
        alpha_bar_t = diffusion.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_next = diffusion.alpha_bars[t_next].view(-1, 1, 1, 1)
        x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        x = torch.sqrt(alpha_bar_next) * x_0_pred + torch.sqrt(1 - alpha_bar_next) * pred_noise
        x = torch.clamp(x, -1, 1)

        if i % 20 == 0:
            tqdm.write(f"Step {i}: min={x.min().item():.4f}, max={x.max().item():.4f}")

    print("Raw generated images min:", x.min().item(), "max:", x.max().item())
    return x

# 主程序
if __name__ == "__main__":
    model_path = "models/cond_hrd_style_mnist_mlp_s.pth"
    digits = list(range(10))
    num_styles_per_digit = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "images/generated_grid_mnist_mlp_s.png"

    model = ConditionalDiffusionModel(
        input_dim=1,
        output_dim=1,
        num_classes=10,
        num_styles_per_class=num_styles_per_digit,
        time_dim=128,
        channels=[32, 64, 128, 128, 256]
    ).to(device)
    diffusion = DiffusionProcess(num_timesteps=500, device=device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")

    model.eval()
    all_final_images = []
    with torch.no_grad():
        labels = torch.tensor([i for i in range(10) for _ in range(num_styles_per_digit)], dtype=torch.long, device=device)
        styles = torch.arange(num_styles_per_digit, dtype=torch.long, device=device).repeat(10)
        all_final_images = generate(model, diffusion, labels, styles, device, steps=300)

        print(f"Generated images min: {all_final_images.min().item()}, max: {all_final_images.max().item()}")

        images_np = all_final_images.cpu().numpy()
        for i in range(images_np.shape[0]):
            img = images_np[i, 0]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                images_np[i, 0] = 2 * (img - img_min) / (img_max - img_min) - 1
        images_display = (images_np + 1) / 2
        images_display = np.clip(images_display, 0, 1)
        
        images_display = images_display.reshape(1, *images_display.shape )
        num_digits = len(digits)
        fig, axes = plt.subplots(num_digits, num_styles_per_digit, figsize=(num_styles_per_digit * 2, num_digits * 2))
        for i, digit in enumerate(digits):
            for j in range(num_styles_per_digit):
                idx = i * num_styles_per_digit + j
                axes[i, j].imshow(images_display[idx, 0], cmap="gray", vmin=0, vmax=1)
                axes[i, j].set_title(f"Style {j}")
                if j == 0:
                    axes[i, j].set_ylabel(f"Digit {digit}")
                axes[i, j].axis("off")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Samples saved to {save_path}")
        plt.show()
