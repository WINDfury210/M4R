import os
import json
import torch
import random
from datetime import datetime
from torch.utils.data import Dataset
from model import ConditionalUNet1D

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self._linear_beta_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
    
    def _linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.num_timesteps)
    
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0, device=self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise


class FinancialDataset(Dataset):
    def __init__(self, data_path, scale_factor=1.0):
        data = torch.load(data_path)
        self.sequences = data["sequences"]
        self.dates = data["start_dates"]
        self.original_mean = self.sequences.mean().item()
        self.original_std = self.sequences.std().item()
        self.scale_factor = scale_factor
        self.sequences = (self.sequences - self.original_mean) / self.original_std * scale_factor
        print(f"Scaled data - Mean: {self.sequences.mean().item():.6f}, Std: {self.sequences.std().item():.6f}")
        print(f"Total sequences: {len(self.sequences)}")
        print(f"Sequences shape: {self.sequences.shape}")
        print(f"Dates shape: {self.dates.shape}")
        print(f"Dates[:, 0] range: {self.dates[:, 0].min().item():.6f} to {self.dates[:, 0].max().item():.6f}")
        unique_years = torch.unique(self.dates[:, 0]).tolist()
        print(f"Unique dates[:, 0] values: {unique_years[:10]}{'...' if len(unique_years) > 10 else ''}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "date": self.dates[idx]
        }
    
    def get_annual_start_dates(self, years):
        min_year, max_year = 2017, 2024
        start_dates = []
        for year in years:
            norm_year = (year - min_year) / 8.0
            start_date = torch.tensor([norm_year, 0.0, 0.0], dtype=torch.float32)
            start_dates.append(start_date)
        return torch.stack(start_dates)
    
    def get_random_dates_for_year(self, year, num_samples):
        min_year, max_year = 2017, 2024
        norm_year = (year - min_year) / 8.0
        random_dates = []
        for _ in range(num_samples):
            norm_month = random.uniform(0, 1)
            norm_day = random.uniform(0, 1)
            random_date = torch.tensor([norm_year, norm_month, norm_day], dtype=torch.float32)
            random_dates.append(random_date)
        return torch.stack(random_dates)
    
    def inverse_scale(self, sequences):
        return sequences * self.original_std / self.scale_factor + self.original_mean

@torch.no_grad()
def generate_samples(model, diffusion, condition, num_samples, device, steps=1000, step_interval=50):
    model.eval()
    labels = condition["date"].to(device)
    x = torch.randn(num_samples, 256, device=device)
    intermediate_samples = {}
    step_indices = torch.linspace(diffusion.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    
    # Determine target timesteps for saving (every step_interval)
    target_ts = list(range(0, diffusion.num_timesteps + 1, step_interval))[::-1]
    
    for i in range(steps):
        t = step_indices[i]
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, labels)
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        alpha_t = diffusion.alphas[t].view(-1, 1)
        beta_t = diffusion.betas[t].view(-1, 1)
        x = (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar * pred_noise) / torch.sqrt(alpha_t)
        if t > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)
        
        # Save intermediate samples at target timesteps
        if int(t+1) in target_ts:
            intermediate_samples[int(t+1)] = x.cpu()
    
    return x.cpu(), intermediate_samples

class SequenceGenerator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DiffusionProcess(device=self.device)
        self.model = self._load_model()
        self.dataset = FinancialDataset(config["data_path"])
        
    def _load_model(self):
        model = ConditionalUNet1D(seq_len=256, channels=self.config["channels"]).to(self.device)
        checkpoint = torch.load(self.config["model_path"], map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    
    def generate_for_year(self, year, num_samples):
        random_dates = self.dataset.get_random_dates_for_year(year, num_samples)
        all_sequences = []
        all_intermediate = {}  # 保存所有样本的中间结果
        
        for i in range(num_samples):
            condition = {"date": random_dates[i].unsqueeze(0).to(self.device)}
            gen_data, intermediate_samples = generate_samples(
                self.model, self.diffusion, condition,
                num_samples=1,
                device=self.device,
                steps=self.config["diffusion_steps"],
                step_interval=self.config.get("step_interval", 50)
            )
            gen_data = self.dataset.inverse_scale(gen_data)
            all_sequences.append(gen_data.squeeze().cpu())
            
            # 合并中间结果
            for t, samples in intermediate_samples.items():
                if t not in all_intermediate:
                    all_intermediate[t] = []
                all_intermediate[t].append(samples.squeeze(0))
        
        # 将中间结果转换为张量
        for t in all_intermediate:
            all_intermediate[t] = torch.stack(all_intermediate[t])
        
        return torch.stack(all_sequences), all_intermediate
    
    def save_sequences(self, sequences, intermediate_samples, year, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        data = {
            "sequences": sequences,
            "intermediate_samples": intermediate_samples,  # 保存中间时间步数据
            "metadata": {
                "year": year,
                "num_samples": len(sequences),
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "step_interval": self.config.get("step_interval", 50)
            }
        }
        filename = os.path.join(output_dir, f"generated_{year}.pt")
        torch.save(data, filename)
        return filename

def main():
    config = {
        "model_path": "saved_models/final_model.pth",
        "data_path": "financial_data/sequences/sequences_256.pt",
        "channels": [32, 128, 512, 2048],
        "years": list(range(2017, 2024)),
        "samples_per_year": 100,
        "diffusion_steps": 1000,
        "step_interval": 50,  # 新增：控制中间结果保存间隔
        "output_dir": "generated_sequences"
    }
    
    # 固定随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"generation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = SequenceGenerator(config)
    
    # Generate sequences for each year
    for year in config["years"]:
        print(f"Generating {config['samples_per_year']} samples for year {year}...")
        sequences, intermediate_samples = generator.generate_for_year(
            year, 
            config["samples_per_year"]
        )
        save_path = generator.save_sequences(
            sequences, 
            intermediate_samples, 
            year, 
            output_dir
        )
        print(f"Saved to {save_path}")
    
    # Save config for reference
    config_path = os.path.join(output_dir, "generation_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nGeneration complete! Results saved to {output_dir}")

if __name__ == "__main__":
    import numpy as np  # 需要添加的导入
    main()