import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# 定义时间嵌入模块
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0, dtype=torch.float32)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb)

    def forward(self, time):
        emb = time[:, None] * self.emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

# 定义条件 UNet 模型（适配 28x28）
class ConditionalUNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, num_styles_per_class=5, time_dim=128, channels=[64, 128, 256, 512]):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.label_embedding = nn.Embedding(num_classes, time_dim)
        self.style_embeddings = nn.ModuleList([nn.Embedding(num_styles_per_class, time_dim) for _ in range(num_classes)])

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

# 定义数据增强
data_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
])

# 定义训练函数
def train(model, dataloader, diffusion, optimizer, scheduler, device, num_epochs=1000):
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0
        for x, labels, styles in dataloader:
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)
            styles = styles.to(device)

            x = data_transforms(x)

            optimizer.zero_grad()
            t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=device)
            noisy_x, noise = diffusion.add_noise(x, t)
            pred_noise = model(noisy_x, t, labels, styles)
            loss = F.mse_loss(pred_noise, noise)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), "models/cond_hrd_style_mnist_small.pth")
    print("Model saved to models/cond_hrd_style_mnist.pth")

# 加载 MNIST 数据并进行聚类
def load_and_cluster_mnist_data(num_classes=10, num_clusters_per_class=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 从 train_dataset 获取处理后的数据
    X = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])  # 形状: (60000, 1, 28, 28)
    y = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])  # 形状: (60000,)
    print(f"Normalized X range: min={X.min().item():.4f}, max={X.max().item():.4f}")
    # 聚类生成风格标签
    style_labels = torch.zeros_like(y)
    X_flat = X.view(len(X), -1).numpy()  # 展平为 (60000, 784)
    for cls in range(num_classes):
        cls_mask = (y == cls).numpy()
        cls_data = X_flat[cls_mask]
        if len(cls_data) > num_clusters_per_class:
            kmeans = KMeans(n_clusters=num_clusters_per_class, random_state=42).fit(cls_data)
            style_labels[cls_mask] = torch.tensor(kmeans.labels_, dtype=torch.long)

    return TensorDataset(X, y, style_labels)

# 主程序
if __name__ == "__main__":
    dataset = load_and_cluster_mnist_data()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ConditionalDiffusionModel(
        input_dim=1,
        output_dim=1,
        num_classes=10,
        num_styles_per_class=5,
        time_dim=128,
        channels=[64, 128]
    ).to(device)
    diffusion = DiffusionProcess(num_timesteps=500, device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    train(model, dataloader, diffusion, optimizer, scheduler, device, num_epochs=500)