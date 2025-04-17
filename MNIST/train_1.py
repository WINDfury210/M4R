import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# 定义时间嵌入模块
class TimeEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, time):
        time = time.unsqueeze(-1).float()
        emb = self.mlp(time)
        return emb

# 定义条件 UNet 模型（移除初始化）
class ConditionalUNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, time_dim=128, channels=[64, 128, 256, 512]):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.label_embedding = nn.Embedding(num_classes, time_dim)

        self.channels = channels
        self.num_layers = len(channels)

        self.encoder_convs = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        in_channels = input_dim
        for out_channels in channels:
            self.encoder_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.encoder_norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

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

    def forward(self, x, time, labels):
        time_emb = self.time_embedding(time)
        label_emb = self.label_embedding(labels)

        time_emb = self.fc_time(time_emb).view(-1, self.channels[-1], 1, 1)
        label_emb = self.fc_label(label_emb).view(-1, self.channels[-1], 1, 1)

        skips = []
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            residual = x
            x = F.relu(norm(conv(x)))
            if residual.shape[1] == x.shape[1]:
                x = x + residual
            skips.append(x)

        x = x + time_emb + label_emb

        for i, (conv, norm) in enumerate(zip(self.decoder_convs, self.decoder_norms)):
            skip = skips[-2 - i]
            x = torch.cat([x, skip], dim=1)
            residual = x
            x = F.relu(norm(conv(x)))
            if residual.shape[1] == x.shape[1]:
                x = x + residual
        x = torch.cat([x, skips[0]], dim=1)
        x = self.final_conv(x) * 0.1

        return x

# 定义条件扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, time_dim=128, channels=[64, 128, 256, 512]):
        super().__init__()
        self.unet = ConditionalUNet(input_dim, output_dim, num_classes, time_dim, channels)

    def forward(self, x, time, labels):
        return self.unet(x, time, labels)

# 定义 VP-SDE 扩散过程
class DiffusionProcess:
    def __init__(self, num_timesteps=500, beta_min=0.0001, beta_max=0.01, schedule="cosine", device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        t = torch.linspace(0, 1, num_timesteps + 1, device=device, dtype=torch.float32)[:-1]
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_min, beta_max, num_timesteps, device=device)
            integrals = torch.cumsum(self.betas * (1.0 / num_timesteps), dim=0)
            self.alphas = torch.exp(-0.5 * integrals)
            self.sigma_squares = 1 - torch.exp(-integrals)
        elif schedule == "cosine":
            self.sigma_squares = 1 - torch.cos(np.pi * t)
            self.alphas = 1 - self.sigma_squares
            self.betas = torch.diff(self.sigma_squares, prepend=torch.tensor([0.0], device=device)) / (1.0 / num_timesteps)
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
            x = x + sigma_t * torch.randn_like(x)
        elif method == "pc":
            x = x + (f_t - g_t**2 * s_theta) * dt + g_t * torch.sqrt(torch.abs(dt)) * torch.randn_like(x)
            if lambda_corrector > 0:
                t_next_tensor = torch.full((labels.size(0),), t_next, device=device)
                pred_noise_next = model(x, t_next_tensor, labels)
                s_theta_next = -pred_noise_next / sigma_t_next
                x = x + lambda_corrector * s_theta_next + torch.sqrt(2 * lambda_corrector) * torch.randn_like(x)
        
        if clamp_range:
            x = torch.clamp(x, *clamp_range)
        
        if i % 20 == 0:
            tqdm.write(f"Step {i}: min={x.min().item():.4f}, max={x.max().item():.4f}")
    
    return x

# 定义训练函数（严格恢复原始逻辑）
def train(model, dataloader, diffusion, optimizer, scheduler, device, num_epochs=250):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        for x, labels, _ in dataloader:
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()
            t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=device)
            noisy_x, noise = diffusion.add_noise(x, t)
            pred_noise = model(noisy_x, t, labels)
            loss = F.mse_loss(pred_noise, noise)

            if torch.isnan(loss) or torch.isinf(loss):
                continue  # 跳过 NaN loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cond_mnist_vp.pth")
    print("Model saved to models/cond_mnist_vp.pth")

# 加载 MNIST 数据
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    X = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    print(f"Normalized X range: min={X.min().item():.4f}, max={X.max().item():.4f}")
    
    return TensorDataset(X, y, torch.zeros_like(y))

# 主程序
if __name__ == "__main__":
    dataset = load_mnist_data()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionModel(
        input_dim=1,
        output_dim=1,
        num_classes=10,
        time_dim=128,
        channels=[64, 64, 128, 128, 256]
    ).to(device)
    diffusion = DiffusionProcess(num_timesteps=500, beta_min=0.0001, beta_max=0.01, schedule="cosine", device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    train(model, dataloader, diffusion, optimizer, scheduler, device, num_epochs=250)

    model.eval()
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

    all_images = {}
    with torch.no_grad():
        for method_config in sampling_methods:
            method = method_config["method"]
            images = generate(
                model, diffusion, labels, device, input_shape,
                steps=100, method=method, eta=method_config["eta"],
                lambda_corrector=method_config["lambda_corrector"],
                clamp_range=(-1, 1)
            )
            all_images[method] = images.cpu().numpy()
            print(f"Generated {method} images: min={images.min().item():.4f}, max={images.max().item():.4f}")

    for method, images_np in all_images.items():
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