import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import stats
from statsmodels.tsa.stattools import acf

# 1. Data Loading ------------------------------------------------------------

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
    
    def get_all_sequences_for_year(self, year, max_samples=4197):
        min_year, max_year = 2017, 2024
        norm_year = (year - min_year) / 8.0  # 匹配数据生成公式
        year_mask = torch.abs(self.dates[:, 0] - norm_year) < 1e-3
        year_indices = torch.where(year_mask)[0]
        print(f"Year {year}: Found {len(year_indices)} sequences")
        if len(year_indices) == 0:
            print(f"Warning: No sequences found for year {year}")
            return torch.zeros(0, 256), torch.tensor([])
        if len(year_indices) > max_samples:
            random_indices = torch.randperm(len(year_indices))[:max_samples]
            year_indices = year_indices[random_indices]
        return self.sequences[year_indices], year_indices
    
    def inverse_scale(self, sequences):
        if sequences.numel() == 0:
            return sequences
        return sequences * self.original_std / self.scale_factor + self.original_mean

# 2. Metrics Calculation -----------------------------------------------------

def calculate_metrics(real_data):
    metrics = {}
    
    if real_data.numel() == 0 or real_data.dim() == 0:
        print("Warning: Empty or invalid real_data, returning default metrics")
        return {
            'real_mean': 0.0, 'real_std': 0.0, 'real_corr': 0.0, 'real_acf': 0.0,
            'real_skew': 0.0, 'real_kurt': 0.0,
            'abs_real_mean': 0.0, 'abs_real_std': 0.0, 'abs_real_corr': 0.0, 'abs_real_acf': 0.0,
            'abs_real_skew': 0.0, 'abs_real_kurt': 0.0
        }
    
    if real_data.dim() == 1:
        real_data = real_data.unsqueeze(0)
    
    metrics['real_mean'] = real_data.mean().item()
    metrics['real_std'] = real_data.std().item()
    
    real_data_np = real_data.cpu().numpy().flatten()
    real_lagged = real_data_np[:-1]
    real_next = real_data_np[1:]
    metrics['real_corr'] = np.corrcoef(real_lagged, real_next)[0, 1] if len(real_lagged) > 1 else 0.0
    
    try:
        real_acf = acf(real_data_np, nlags=20, fft=True)[1:].mean()
    except Exception as e:
        print(f"Warning: ACF computation failed, setting real_acf to 0.0: {e}")
        real_acf = 0.0
    metrics['real_acf'] = real_acf
    
    metrics['real_skew'] = stats.skew(real_data_np)
    metrics['real_kurt'] = stats.kurtosis(real_data_np)
    
    abs_real_data = torch.abs(real_data)
    abs_real_data_np = abs_real_data.cpu().numpy().flatten()
    
    metrics['abs_real_mean'] = abs_real_data.mean().item()
    metrics['abs_real_std'] = abs_real_data.std().item()
    
    abs_real_lagged = abs_real_data_np[:-1]
    abs_real_next = abs_real_data_np[1:]
    metrics['abs_real_corr'] = np.corrcoef(abs_real_lagged, abs_real_next)[0, 1] if len(abs_real_lagged) > 1 else 0.0
    
    try:
        abs_real_acf = acf(abs_real_data_np, nlags=20, fft=True)[1:].mean()
    except Exception as e:
        print(f"Warning: Abs ACF computation failed, setting abs_real_acf to 0.0: {e}")
        abs_real_acf = 0.0
    metrics['abs_real_acf'] = abs_real_acf
    
    metrics['abs_real_skew'] = stats.skew(abs_real_data_np)
    metrics['abs_real_kurt'] = stats.kurtosis(abs_real_data_np)
    
    return metrics

def compute_stats(metrics_list):
    if not metrics_list:
        print("Warning: Empty metrics_list, returning default stats")
        return {
            'real_mean': {'mean': 0.0, 'variance': 0.0},
            'real_std': {'mean': 0.0, 'variance': 0.0},
            'real_corr': {'mean': 0.0, 'variance': 0.0},
            'real_acf': {'mean': 0.0, 'variance': 0.0},
            'real_skew': {'mean': 0.0, 'variance': 0.0},
            'real_kurt': {'mean': 0.0, 'variance': 0.0},
            'abs_real_mean': {'mean': 0.0, 'variance': 0.0},
            'abs_real_std': {'mean': 0.0, 'variance': 0.0},
            'abs_real_corr': {'mean': 0.0, 'variance': 0.0},
            'abs_real_acf': {'mean': 0.0, 'variance': 0.0},
            'abs_real_skew': {'mean': 0.0, 'variance': 0.0},
            'abs_real_kurt': {'mean': 0.0, 'variance': 0.0}
        }
    
    stats = {}
    keys = list(metrics_list[0].keys())
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            stats[key] = {
                'mean': float(np.mean(values)),
                'variance': float(np.var(values))
            }
        else:
            stats[key] = {
                'mean': 0.0,
                'variance': 0.0
            }
    return stats

# 3. Report Generation -------------------------------------------------------

def print_real_metrics_report(metrics_dict, years):
    print("\n=== Real Data Metrics Report ===")
    
    print("\n[Global Statistics]")
    print(f"{'Metric':<20} {'Mean':>12} {'Variance':>12}")
    print("-" * 44)
    global_stats = metrics_dict.get('global', {})
    for metric in ['real_mean', 'real_std', 'real_corr', 'real_acf', 'real_skew', 'real_kurt']:
        mean = global_stats.get(metric, {}).get('mean', 0.0)
        variance = global_stats.get(metric, {}).get('variance', 0.0)
        print(f"{metric:<20} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Absolute Value Global Statistics]")
    print(f"{'Metric':<20} {'Mean':>12} {'Variance':>12}")
    print("-" * 44)
    for metric in ['abs_real_mean', 'abs_real_std', 'abs_real_corr', 'abs_real_acf', 'abs_real_skew', 'abs_real_kurt']:
        mean = global_stats.get(metric, {}).get('mean', 0.0)
        variance = global_stats.get(metric, {}).get('variance', 0.0)
        print(f"{metric:<20} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Yearly Statistics]")
    print(f"{'Year':<6} {'Metric':<20} {'Mean':>12} {'Variance':>12}")
    print("-" * 50)
    for year in years:
        year_stats = metrics_dict.get(f'year_{year}', {})
        print(f"Year {year} stats keys: {list(year_stats.keys())}")
        for metric in ['real_mean', 'real_std', 'real_corr', 'real_acf', 'real_skew', 'real_kurt']:
            mean = year_stats.get(metric, {}).get('mean', 0.0)
            variance = year_stats.get(metric, {}).get('variance', 0.0)
            print(f"{year:<6} {metric:<20} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Absolute Value Yearly Statistics]")
    print(f"{'Year':<6} {'Metric':<20} {'Mean':>12} {'Variance':>12}")
    print("-" * 50)
    for year in years:
        year_stats = metrics_dict.get(f'year_{year}', {})
        for metric in ['abs_real_mean', 'abs_real_std', 'abs_real_corr', 'abs_real_acf', 'abs_real_skew', 'abs_real_kurt']:
            mean = year_stats.get(metric, {}).get('mean', 0.0)
            variance = year_stats.get(metric, {}).get('variance', 0.0)
            print(f"{year:<6} {metric:<20} {mean:>12.6f} {variance:>12.6f}")

# 4. Main Function -----------------------------------------------------------

def compute_real_metrics(config):
    dataset = FinancialDataset(config["data_path"], scale_factor=1.0)
    years = list(range(2017, 2025))
    metrics = {}
    all_real_metrics = []
    
    output_dir = config["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    for year in years:
        real_seqs, _ = dataset.get_all_sequences_for_year(year, max_samples=config.get("max_samples", 4197))
        real_seqs = dataset.inverse_scale(real_seqs)
        print(f"Year {year}: {len(real_seqs)} real sequences processed")
        
        year_metrics_list = []
        for i in range(len(real_seqs)):
            real_data = real_seqs[i].unsqueeze(0)
            real_metrics = calculate_metrics(real_data)
            year_metrics_list.append(real_metrics)
        
        if not year_metrics_list:
            print(f"Warning: No metrics computed for year {year}")
            metrics[f'year_{year}'] = compute_stats([])
            continue
        
        print(f"Year {year}: Sample metrics keys: {list(year_metrics_list[0].keys())}")
        
        year_stats = compute_stats(year_metrics_list)
        metrics[f'year_{year}'] = year_stats
        
        with open(os.path.join(output_dir, f'real_metrics_{year}.json'), 'w') as f:
            json.dump(year_stats, f, indent=2)
        
        all_real_metrics.extend(year_metrics_list)
    
    global_stats = compute_stats(all_real_metrics)
    metrics['global'] = global_stats
    
    with open(os.path.join(output_dir, 'real_metrics_global.json'), 'w') as f:
        json.dump(global_stats, f, indent=2)
    
    print_real_metrics_report(metrics, years)
    
    print(f"\nMetrics saved to {output_dir}")
    print(f"Computation complete!")

if __name__ == "__main__":
    config = {
        "data_path": "financial_data/sequences/sequences_256.pt",
        "save_dir": "real_metrics",
        "max_samples": 10000
    }
    compute_real_metrics(config)