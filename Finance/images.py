import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
output_dir = "financial_data/figures"
os.makedirs(output_dir, exist_ok=True)
input_file = "financial_data/sequences/sequences_256.pt"
output_file = os.path.join(output_dir, "sample_sequences.png")

# 加载数据
try:
    data = torch.load(input_file)
    sequences = data["sequences"].numpy()
    start_dates = data["start_dates"].numpy()
    logger.info(f"Loaded data: {sequences.shape} sequences, {start_dates.shape} start dates")
except Exception as e:
    logger.error(f"Failed to load data from {input_file}: {e}")
    raise

# 选择示例序列
num_samples = min(5, len(sequences))  # 最多展示5个序列
sample_indices = np.random.choice(len(sequences), num_samples, replace=False)
sample_sequences = sequences[sample_indices]
sample_dates = start_dates[sample_indices]

# 绘制序列
plt.figure(figsize=(12, 6))
for i in range(num_samples):
    plt.plot(range(256), sample_sequences[i], label=f'Sequence {i+1} (Start: {sample_dates[i]})')

plt.title('Sample Daily Log-Return Sequences (256 Days)')
plt.xlabel('Day')
plt.ylabel('Log-Return')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图片
plt.savefig(output_file, dpi=300)
logger.info(f"Saved sample sequences plot to {output_file}")
plt.close()