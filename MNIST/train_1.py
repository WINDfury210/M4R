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
    def __init__(self, dim, embedding_type="sinusoidal", hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.embedding_type = embedding_type
        if embedding_type == "sinusoidal":
            half_dim = dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            self.register_buffer("emb", emb)
        elif embedding_type == "linear":
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )

    def forward(self, time):
        if self.embedding_type == "sinusoidal":
            emb = time[:, None] * self.emb[None, :]
            emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1, 0, 0))
            return emb
        elif self.embedding_type == "linear":
            time = time.unsqueeze(-1).float()
            return self.mlp(time)

# 定义自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, ch, h, w = x.size()
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)
        attn = self.softmax(torch.bmm(q, k) / (ch // 8) ** 0.5)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch, ch, h, w)
        return x + self.gamma * out

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

# 定义条件 UNet 模型
class ConditionalUNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, time_dim=512, channels=[32, 64, 128, 256]):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, embedding_type="sinusoidal")
        self.label_embedding = nn.Embedding(num_classes, time_dim)
        self.dropout = nn.Dropout(p=0.03)

        self.channels = channels
        self.num_layers = len(channels)

        self.encoder_convs = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_channels = input_dim
        for i, out_channels in enumerate(channels):
            self.encoder_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.encoder_res.append(ResidualBlock(out_channels, out_channels))
            self.attentions.append(SelfAttention(out_channels) if i in [1, 2] else nn.Identity())  # 在 channels=64, 128 添加注意力
            in_channels = out_channels

        self.decoder_convs = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        for i in range(len(channels) - 1):
            in_channels = channels[-1 - i] + channels[-2 - i]
            out_channels = channels[-2 - i]
            self.decoder_convs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.decoder_res.append(ResidualBlock(out_channels, out_channels))
        self.final_conv = nn.ConvTranspose2d(channels[0] + channels[0], output_dim, kernel_size=3, padding=1)

        self.fc_time = nn.Linear(time_dim, channels[-1])
        self.fc_label = nn.Linear(time_dim, channels[-1])

    def forward(self, x, time, labels):
        time_emb = self.time_embedding(time)
        label_emb = self.label_embedding(labels)
        time_emb = self.fc_time(time_emb).view(-1, self.channels[-1], 1, 1)
        label_emb = self.fc_label(label_emb).view(-1, self.channels[-1], 1, 1)

        skips = []
        for i, (conv, res, attn) in enumerate(zip(self.encoder_convs, self.encoder_res, self.attentions)):
            x = F.relu(conv(x))
            x = res(x)
            x = attn(x)
            x = self.dropout(x)
            skips.append(x)

        x = x + time_emb + label_emb

        for i, (conv, res) in enumerate(zip(self.decoder_convs, self.decoder_res)):
            skip = skips[-2 - i]
            x = torch.cat([x, skip], dim=1)
            x = F.relu(conv(x))
            x = res(x)
            x = self.dropout(x)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.final_conv(x)

        return x

# 定义扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, time_dim=512, channels=[32, 64, 128, 256]):
        super().__init__()
        self.unet = ConditionalUNet(input_dim, output_dim, num_classes, time_dim, channels)

    def forward(self, x, time, labels):
        return self.unet(x, time, labels)

# 定义余弦调度
def cosine_beta_schedule(num_timesteps):
    s = 0.008
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# 定义扩散过程
class DiffusionProcess:
    def __init__(self, num_timesteps=200, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = cosine_beta_schedule(num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def add_noise(self, x, t, clamp_range=(-1, 1)):
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x, device=self.device)
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        if clamp_range:
            noisy_x = torch.clamp(noisy_x, *clamp_range)
        return noisy_x, noise

# 生成函数
@torch.no_grad()
def generate(model, diffusion, labels, device, input_shape, steps=1500, method="ddim", eta=0.0, lambda_corrector=0.0, clamp_range=(-1, 1)):
    model.eval()
    x = torch.randn((labels.size(0), *input_shape), device=device, dtype=torch.float32)
    step_indices = torch.linspace(diffusion.num_timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)
    
    for i in tqdm(range(steps), desc=f"Generating Steps ({method})"):
        t = step_indices[i]
        t_next = step_indices[i + 1]
        t_tensor = torch.full((labels.size(0),), t, device=device)
        pred_noise = model(x, t_tensor, labels)
        
        if i % 200 == 0:
            print(f"Step {i}: pred_noise mean={pred_noise.mean().item():.4f}, std={pred_noise.std().item():.4f}")
        
        sqrt_alpha_bar = diffusion.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_alpha_bar_next = diffusion.sqrt_alpha_bars[t_next].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_next = diffusion.sqrt_one_minus_alpha_bars[t_next].view(-1, 1, 1, 1)
        alpha_t = diffusion.alphas[t].view(-1, 1, 1, 1)
        beta_t = diffusion.betas[t].view(-1, 1, 1, 1)
        
        if method == "ddim":
            x_0_pred = (x - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            x = sqrt_alpha_bar_next * x_0_pred + sqrt_one_minus_alpha_bar_next * pred_noise
        elif method == "hybrid":
            x_0_pred = (x - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
            x = sqrt_alpha_bar_next * x_0_pred + sqrt_one_minus_alpha_bar_next * pred_noise + eta * sqrt_one_minus_alpha_bar_next * torch.randn_like(x)
        elif method == "ddpm":
            x = (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(alpha_t)
            if t > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        elif method == "pc":
            s_theta = -pred_noise / sqrt_one_minus_alpha_bar
            f_t = -0.5 * beta_t * x
            g_t = torch.sqrt(beta_t)
            dt = torch.tensor(-1.0 / diffusion.num_timesteps, device=device)
            x = x + (f_t - g_t**2 * s_theta) * dt + g_t * torch.sqrt(torch.abs(dt)) * torch.randn_like(x)
            if lambda_corrector > 0 and t_next > 0:
                t_next_tensor = torch.full((labels.size(0),), t_next, device=device)
                pred_noise_next = model(x, t_next_tensor, labels)
                s_theta_next = -pred_noise_next / sqrt_one_minus_alpha_bar_next
                x = x + lambda_corrector * s_theta_next + torch.sqrt(torch.tensor(2 * lambda_corrector, device=device)) * torch.randn_like(x)
        
        if clamp_range:
            x = torch.clamp(x, *clamp_range)
        
        if i % 200 == 0:
            images_display = (x.cpu().numpy() + 1) / 2
            images_display = np.clip(images_display, 0, 1)
            os.makedirs("images", exist_ok=True)
            for j in range(min(3, x.size(0))):  # 保存前 3 张
                plt.imshow(images_display[j, 0], cmap="gray")
                plt.axis("off")
                plt.savefig(f"images/intermediate_mnist_step_{i}_sample_{j+1}_{method}.png")
                plt.close()
    
    return x

# 定义数据增强
data_transforms = transforms.Compose([
    transforms.RandomRotation(3),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)
])

# 定义训练函数
def train(model, dataloader, diffusion, optimizer, scheduler, device, num_epochs=4000):
    model.train()
    loss_history = []
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0
        for x, labels in dataloader:
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)
            x = data_transforms(x)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=device)
            noisy_x, noise = diffusion.add_noise(x, t)
            pred_noise = model(noisy_x, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        if (epoch + 1) % 1000 == 0:
            torch.save(model.state_dict(), f"models/cond_mnist_mlp_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), "models/cond_mnist_mlp_final.pth")
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.savefig("images/loss_curve.png")
    plt.close()
    print("Model saved to models/cond_mnist_mlp_final.pth")
    return loss_history

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
    return TensorDataset(X, y)

# 主程序
if __name__ == "__main__":
    dataset = load_mnist_data()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDiffusionModel(
        input_dim=1,
        output_dim=1,
        num_classes=10,
        time_dim=512,
        channels=[32, 64, 128, 256]
    ).to(device)
    diffusion = DiffusionProcess(num_timesteps=200, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000, eta_min=1e-6)
    loss_history = train(model, dataloader, diffusion, optimizer, scheduler, device, num_epochs=4000)

    model.eval()
    num_samples_per_digit = 10
    digits = list(range(10))
    labels = torch.tensor([i for i in digits for _ in range(num_samples_per_digit)], dtype=torch.long, device=device)
    input_shape = (1, 28, 28)
    sampling_methods = [
        {"method": "ddim", "eta": 0.0, "lambda_corrector": 0.0},
        {"method": "ddpm", "eta": 0.0, "lambda_corrector": 0.0},
        {"method": "hybrid", "eta": 0.05, "lambda_corrector": 0.0},
        {"method": "pc", "eta": 0.0, "lambda_corrector": 0.0}
    ]

    with torch.no_grad():
        # 仅运行 DDIM
        method_config = sampling_methods[0]  # DDIM
        method = method_config["method"]
        try:
            images = generate(
                model, diffusion, labels, device, input_shape,
                steps=1500, method=method, eta=method_config["eta"],
                lambda_corrector=method_config["lambda_corrector"],
                clamp_range=(-1, 1)
            )
            images_np = images.cpu().numpy()
            print(f"Generated {method} images: min={images.min().item():.4f}, max={images.max().item():.4f}")

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
        
        except Exception as e:
            print(f"Error in {method} sampling: {str(e)}")