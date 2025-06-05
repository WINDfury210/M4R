import os
import json
import torch
import random
from datetime import datetime
from torch.utils.data import Dataset
from model import ConditionalUNet1D
import numpy as np

class DiffusionProcess:
    def __init__(self, num_timesteps=1000, device="cpu", beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self._beta_schedule(beta_schedule).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1. - self.alpha_bars)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_betas = torch.sqrt(self.betas)
        self.one_minus_alphas = 1. - self.alphas

    def _beta_schedule(self, schedule_type):
        if schedule_type == "linear":
            return torch.linspace(1e-4, 0.02, self.num_timesteps)
        
        elif schedule_type == "cosine":
            steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32) / self.num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            betas = torch.clamp(1. - alpha_bar[1:] / alpha_bar[:-1], min=1e-4, max=0.999)
            return betas
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")

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
        print(f"Total sequences: {len(self.sequences)}, Shape: {self.sequences.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx], "date": self.dates[idx]}

    def get_annual_start_dates(self, years):
        min_year, max_year = 2017, 2024
        start_dates = [torch.tensor([(year - min_year) / 8.0, 0.0, 0.0], dtype=torch.float32) for year in years]
        return torch.stack(start_dates)

    def get_random_dates_for_year(self, year, num_samples):
        min_year, max_year = 2017, 2024
        norm_year = (year - min_year) / 8.0
        random_dates = torch.tensor([[norm_year, random.uniform(0, 1), random.uniform(0, 1)] for _ in range(num_samples)], dtype=torch.float32)
        return random_dates

    def inverse_scale(self, sequences):
        return sequences * self.original_std / self.scale_factor + self.original_mean

@torch.no_grad()
def generate_samples(model, diffusion, condition, num_samples, device, steps=500, step_interval=50):
    model.eval()
    labels = condition["date"].to(device)  # [num_samples, 3]
    x = torch.randn(num_samples, 256, device=device)
    intermediate_samples = {}
    step_indices = torch.linspace(diffusion.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
    target_ts = set(range(0, diffusion.num_timesteps + 1, step_interval)[::-1])

    for t in step_indices:
        t_tensor = torch.full((num_samples,), t.item(), device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, labels)
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alpha_bars[t].view(-1, 1)

        x = (x - diffusion.one_minus_alphas[t].view(-1, 1) / sqrt_one_minus_alpha_bar * pred_noise) / diffusion.sqrt_alphas[t]+ diffusion.sqrt_betas[t].view(-1, 1) * torch.randn_like(x)
        if int(t+1) in target_ts:
            intermediate_samples[int(t+1)] = x.clone().cpu()

    intermediate_samples[0] = x.clone().cpu()
    return x.cpu(), intermediate_samples

class SequenceGenerator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = DiffusionProcess(num_timesteps=config["diffusion_steps"], device=self.device, beta_schedule="linear")
        self.model = self._load_model()
        self.dataset = FinancialDataset(config["data_path"])

    def _load_model(self):
        model = ConditionalUNet1D(seq_len=256, channels=self.config["channels"]).to(self.device)
        checkpoint = torch.load(self.config["model_path"], map_location=self.device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()
        return model

    def generate_for_year(self, year, num_samples):
        random_dates = self.dataset.get_random_dates_for_year(year, num_samples)
        condition = {"date": random_dates.to(self.device)}
        sequences, intermediate_samples = generate_samples(
            self.model, self.diffusion, condition, num_samples, self.device,
            steps=self.config["diffusion_steps"], step_interval=self.config["step_interval"]
        )
        sequences = self.dataset.inverse_scale(sequences)
        intermediate_samples = {t: self.dataset.inverse_scale(samples) for t, samples in intermediate_samples.items()}
        
        if 0 in intermediate_samples:
            assert torch.allclose(sequences, intermediate_samples[0], rtol=1e-5), f"Year {year}: sequences != intermediate_samples[0]"
        
        return sequences, intermediate_samples

    def save_sequences(self, sequences, intermediate_samples, year, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        data = {
            "sequences": sequences,
            "intermediate_samples": intermediate_samples,
            "metadata": {
                "year": year,
                "num_samples": len(sequences),
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "step_interval": self.config["step_interval"]
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
        "diffusion_steps": 500,
        "step_interval": 10,
        "output_dir": "generated_sequences"
    }

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"generation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    generator = SequenceGenerator(config)

    for year in config["years"]:
        print(f"Generating {config['samples_per_year']} samples for year {year}...")
        sequences, intermediate_samples = generator.generate_for_year(year, config["samples_per_year"])
        save_path = generator.save_sequences(sequences, intermediate_samples, year, output_dir)
        print(f"Saved to {save_path}")

    with open(os.path.join(output_dir, "generation_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nGeneration complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()