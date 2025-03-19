import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义条件扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, condition_dim, time_dim=128, hidden_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.fc1 = nn.Linear(input_dim + condition_dim + time_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, condition, time):
        time = time.unsqueeze(-1)  # 将时间步从 (batch_size,) 变为 (batch_size, 1)
        time_emb = self.time_embedding(time)  # 时间嵌入
        x = torch.cat([x, condition, time_emb], dim=-1)  # 将条件信息与时间嵌入拼接
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 扩散过程：添加噪声
def add_noise(x, t, num_timesteps):
    alpha = (1 - t / num_timesteps).view(-1, 1)  # 将 alpha 调整为 (batch_size, 1)
    noise = torch.randn_like(x)  # 高斯噪声
    noisy_x = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    return noisy_x, noise

# 训练函数
def train(model, dataloader, optimizer, device, num_timesteps=100, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, condition in dataloader:
            x = x.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, num_timesteps, (x.size(0),), device=device)  # 随机时间步
            noisy_x, noise = add_noise(x, t, num_timesteps)  # 添加噪声
            pred_noise = model(noisy_x, condition, t.float())  # 预测噪声（将时间步作为条件）
            loss = nn.functional.mse_loss(pred_noise, noise)  # 计算损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

# 生成样本
def generate_samples(model, device, condition, num_samples=1000, num_timesteps=100, guidance_scale=2.0):
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, 2, device=device)  # 从高斯分布中采样
        condition = condition.to(device).unsqueeze(0).repeat(num_samples, 1)  # 重复条件
        for t in reversed(range(num_timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.float32)  # 时间步
            # 预测无条件噪声
            pred_noise_uncond = model(x, torch.zeros_like(condition), t_tensor)
            # 预测条件噪声
            pred_noise_cond = model(x, condition, t_tensor)
            # 结合无条件噪声和条件噪声
            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            alpha = (1 - t / num_timesteps).view(-1, 1)  # 将 alpha 调整为 (batch_size, 1)
            x = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)  # 去噪
    return x.cpu().numpy()

# 生成 GMM 数据
def generate_gmm_data(n_samples=1000, means=None, covariances=None):
    np.random.seed(42)
    if means is None:
        means = [np.array([0, 0]), np.array([4, 4])]  # 默认两个高斯分布的均值
    if covariances is None:
        covariances = [np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 2]])]  # 默认协方差矩阵

    X = []
    labels = []
    for mean, cov in zip(means, covariances):
        samples = np.random.multivariate_normal(mean, cov, n_samples)  # 从当前高斯分布生成样本
        X.append(samples)
        labels.extend([mean] * n_samples)  # 为每个样本分配标签（均值向量）

    X = np.vstack(X)  # 将所有样本堆叠成一个数组
    labels = np.vstack(labels)  # 将标签堆叠成一个数组
    return torch.tensor(X, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# 主程序
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自定义均值和协方差
    means = [np.array([0, 0]), np.array([4, 4])]
    covariances = [np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 2]])]

    # 生成数据
    X, labels = generate_gmm_data(n_samples=1000, means=means, covariances=covariances)
    dataset = torch.utils.data.TensorDataset(X, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    # 初始化模型和优化器
    model = ConditionalDiffusionModel(input_dim=2, condition_dim=2).to(device)  # 条件维度为 2（均值向量）
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    train(model, dataloader, optimizer, device, num_timesteps=100, num_epochs=10)

    # 生成样本（指定条件和引导强度）
    condition = torch.tensor([4.0, 4.0], dtype=torch.float32)  # 假设标签为 [4.0, 4.0]
    guidance_scales = [0.0, 1.0, 2.0, 5.0]  # 不同的引导强度
    for guidance_scale in guidance_scales:
        samples = generate_samples(model, device, condition, num_samples=1000, guidance_scale=guidance_scale)
        plt.figure(figsize=(10, 5))
        plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, label=f"Guidance Scale = {guidance_scale}")
        plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5, label="Original Data")
        plt.legend()
        plt.title(f"Generated Samples with Guidance Scale = {guidance_scale}")
        plt.show()