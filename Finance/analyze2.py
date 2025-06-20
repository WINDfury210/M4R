import os
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf

class FinancialDataset:
    def __init__(self, data_path, scale_factor=1.0):
        data = torch.load(data_path)
        self.sequences = data["sequences"]
        self.dates = data["start_dates"]
        self.original_mean = self.sequences.mean().item()
        self.original_std = self.sequences.std().item()
        self.scale_factor = scale_factor
        self.sequences = (self.sequences - self.original_mean) / self.original_std * scale_factor

    def inverse_scale(self, sequences):
        return sequences * self.original_std / self.scale_factor + self.original_mean

def calculate_metrics(data):
    metrics = {}
    if data.dim() == 1:
        data = data.unsqueeze(0)
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    if data.numel() == 0 or torch.all(data == 0):
        return {key: 0.0 for key in ['gen_mean', 'gen_std', 'gen_corr', 'gen_acf', 'gen_skew', 'gen_kurt',
                                     'abs_gen_mean', 'abs_gen_std', 'abs_gen_corr', 'abs_gen_acf', 'abs_gen_skew', 'abs_gen_kurt']}

    data_np = data.cpu().numpy()
    if data_np.ndim == 1:
        data_np = data_np[np.newaxis, :]

    metrics['gen_mean'] = float(data_np.mean()) if not np.isnan(data_np.mean()) else 0.0
    metrics['gen_std'] = float(data_np.std()) if data_np.size > 1 and not np.isnan(data_np.std()) else 0.0

    sample = data_np[0] if data_np.shape[0] == 1 else data_np.flatten()
    if len(sample) > 1 and np.var(sample) > 1e-10 and not np.isnan(sample).any():
        lagged = sample[:-1]
        next_val = sample[1:]
        metrics['gen_corr'] = np.corrcoef(lagged, next_val)[0, 1] if len(lagged) > 1 else 0.0
        try:
            gen_acf = acf(sample, nlags=20, fft=True)[1:]
            metrics['gen_acf'] = float(gen_acf.mean()) if not np.isnan(gen_acf).all() else 0.0
        except Exception:
            metrics['gen_acf'] = 0.0
    else:
        metrics['gen_corr'] = metrics['gen_acf'] = 0.0

    metrics['gen_skew'] = float(stats.skew(sample)) if len(sample) > 2 and not np.isnan(sample).any() else 0.0
    metrics['gen_kurt'] = float(stats.kurtosis(sample)) if len(sample) > 3 and not np.isnan(sample).any() else 0.0

    abs_data_np = np.abs(data_np)
    metrics['abs_gen_mean'] = float(abs_data_np.mean()) if not np.isnan(abs_data_np.mean()) else 0.0
    metrics['abs_gen_std'] = float(abs_data_np.std()) if abs_data_np.size > 1 and not np.isnan(abs_data_np.std()) else 0.0

    abs_sample = abs_data_np[0] if abs_data_np.shape[0] == 1 else abs_data_np.flatten()
    if len(abs_sample) > 1 and np.var(abs_sample) > 1e-10 and not np.isnan(abs_sample).any():
        abs_lagged = abs_sample[:-1]
        abs_next = abs_sample[1:]
        metrics['abs_gen_corr'] = np.corrcoef(abs_lagged, abs_next)[0, 1] if len(lagged) > 1 else 0.0
        try:
            abs_gen_acf = acf(abs_sample, nlags=20, fft=True)[1:]
            metrics['abs_gen_acf'] = float(abs_gen_acf.mean()) if not np.isnan(abs_gen_acf).all() else 0.0
        except Exception:
            metrics['abs_gen_acf'] = 0.0
    else:
        metrics['abs_gen_corr'] = metrics['abs_gen_acf'] = 0.0

    metrics['abs_gen_skew'] = float(stats.skew(abs_sample)) if len(abs_sample) > 2 and not np.isnan(abs_sample).any() else 0.0
    metrics['abs_gen_kurt'] = float(stats.kurtosis(abs_sample)) if len(abs_sample) > 3 and not np.isnan(abs_sample).any() else 0.0

    return metrics

def average_metrics(metrics_list, store_individual=False):
    if not metrics_list:
        return {key: {'mean': 0.0, 'variance': 0.0} for key in metrics_list[0].keys()} if metrics_list else {}

    stats = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if not np.isnan(m[key])]
        stats[key] = {
            'mean': float(np.mean(values)) if values else 0.0,
            'variance': float(np.var(values)) if len(values) > 1 else 0.0
        }
        if store_individual:
            stats[key]['means'] = values if values else []
    return stats

def save_visualizations(gen_samples, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not gen_samples:
        print(f"Warning: No generated samples for year {year}")
        return

    idx = random.randint(0, len(gen_samples) - 1)
    gen_sample = gen_samples[idx].numpy()
    abs_gen_sample = np.abs(gen_sample)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(gen_sample, label="Generated", color='blue')
    plt.title(f"Generated Sample (Year {year})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    gen_acf = acf(gen_sample, nlags=20, fft=True) if len(gen_sample) > 20 else np.zeros(21)
    plt.stem(range(len(gen_acf)), gen_acf, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title(f"Autocovariance (Year {year})")
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_original_sample.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(abs_gen_sample, label="Abs Generated", color='green')
    plt.title(f"Abs Generated Sample (Year {year})")
    plt.xlabel("Time")
    plt.ylabel("Absolute Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    abs_gen_acf = acf(abs_gen_sample, nlags=20, fft=True) if len(abs_gen_sample) > 20 else np.zeros(21)
    plt.stem(range(len(abs_gen_acf)), abs_gen_acf, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title(f"Abs Autocovariance (Year {year})")
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_absolute.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    stats.probplot(gen_sample, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot (Year {year})")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Prob")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_qq_plot.png'), dpi=300)
    plt.close()

def plot_metrics_vs_timesteps(metrics_per_timestep, output_dir, years, real_metrics_dir="real_metrics"):
    os.makedirs(output_dir, exist_ok=True)
    metrics_to_plot = ['gen_mean', 'gen_std', 'gen_kurt', 'abs_gen_mean', 'abs_gen_std', 'abs_gen_kurt']
    real_metrics_map = {
        'gen_mean': 'real_mean', 'gen_std': 'real_std', 'gen_kurt': 'real_kurt',
        'abs_gen_mean': 'abs_real_mean', 'abs_gen_std': 'abs_real_std', 'abs_gen_kurt': 'abs_real_kurt'
    }

    global_timesteps = sorted(metrics_per_timestep['global'].keys())
    if global_timesteps:
        try:
            with open(os.path.join(real_metrics_dir, 'real_metrics_global.json'), 'r') as f:
                real_global = json.load(f)
        except FileNotFoundError:
            real_global = {}

        plt.figure(figsize=(15, 6))
        for i, metric in enumerate(metrics_to_plot, 1):
            means = [metrics_per_timestep['global'][t].get(metric, {}).get('mean', 0.0) for t in global_timesteps]
            variances = [metrics_per_timestep['global'][t].get(metric, {}).get('variance', 0.0) for t in global_timesteps]

            plt.subplot(2, 3, i)
            plt.plot(global_timesteps[::-1], means, color='blue', label='Generated')
            plt.fill_between(global_timesteps[::-1], [m - np.sqrt(v) for m, v in zip(means, variances)],
                             [m + np.sqrt(v) for m, v in zip(means, variances)], color='blue', alpha=0.2)
            if metric in real_metrics_map:
                real_value = real_global.get(real_metrics_map[metric], {}).get('mean')
                if real_value is not None:
                    plt.axhline(real_value, color='red', linestyle='--', label='Real')
            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Timestep')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_vs_timesteps_global.png'), dpi=300)
        plt.close()

    for year in years:
        if year not in metrics_per_timestep['years']:
            continue
        year_timesteps = sorted(metrics_per_timestep['years'][year].keys())
        if not year_timesteps:
            continue

        try:
            with open(os.path.join(real_metrics_dir, f'real_metrics_{year}.json'), 'r') as f:
                real_year = json.load(f)
        except FileNotFoundError:
            real_year = {}

        plt.figure(figsize=(15, 6))
        for i, metric in enumerate(metrics_to_plot, 1):
            means = [metrics_per_timestep['years'][year][t].get(metric, {}).get('mean', 0.0) for t in year_timesteps]
            variances = [metrics_per_timestep['years'][year][t].get(metric, {}).get('variance', 0.0) for t in year_timesteps]

            plt.subplot(2, 3, i)
            plt.plot(year_timesteps[::-1], means, color='blue', label='Generated')
            plt.fill_between(year_timesteps[::-1], [m - np.sqrt(v) for m, v in zip(means, variances)],
                             [m + np.sqrt(v) for m, v in zip(means, variances)], color='blue', alpha=0.2)
            if metric in real_metrics_map:
                real_value = real_year.get(real_metrics_map[metric], {}).get('mean')
                if real_value is not None:
                    plt.axhline(real_value, color='red', linestyle='--', label='Real')
            plt.title(f"{metric.replace('_', ' ').title()} (Year {year})")
            plt.xlabel('Timestep')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_{year}.png'), dpi=300)
        plt.close()

def print_enhanced_report(metrics_dict, years):
    print("\n=== Validation Report ===")
    print("\n[Global Metrics]")
    print(f"{'Metric':<15} {'Mean':>10} {'Variance':>10}")
    global_stats = metrics_dict.get('global', {})
    for metric in ['gen_mean', 'gen_std', 'gen_kurt', 'abs_gen_mean', 'abs_gen_std', 'abs_gen_kurt']:
        mean = global_stats.get(metric, {}).get('mean', 0.0)
        variance = global_stats.get(metric, {}).get('variance', 0.0)
        print(f"{metric:<15} {mean:>10.6f} {variance:>10.6f}")

    print("\n[Yearly Metrics]")
    print(f"{'Year':<4} {'Metric':<15} {'Mean':>10} {'Variance':>10}")
    for year in years:
        year_stats = metrics_dict.get(f'yearly_{year}')
        if year_stats:
            for metric in ['gen_mean', 'gen_std', 'gen_kurt', 'abs_gen_mean', 'abs_gen_std', 'abs_gen_kurt']:
                mean = year_stats.get(metric, {}).get('mean', 0.0)
                variance = year_stats.get(metric, {}).get('variance', 0.0)
                print(f"{year:<4} {metric:<15} {mean:>10.6f} {variance:>10.6f}")

def validate_generated_data(config):
    years = config.get("years", list(range(2017, 2024)))
    generated_dir = config["generated_dir"]
    output_dir = config["output_dir"]
    real_metrics_dir = config.get("real_metrics_dir", "real_metrics")
    os.makedirs(output_dir, exist_ok=True)

    dataset = FinancialDataset(config["data_path"])
    metrics = {}
    all_gen_samples = []
    metrics_per_timestep = {'global': {}, 'years': {year: {} for year in years}}

    for year in years:
        data_path = os.path.join(generated_dir, f"generated_{year}.pt")
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found, skipping year {year}")
            continue

        print(f"Processing year {year}...")
        data = torch.load(data_path)
        sequences = dataset.inverse_scale(data["sequences"])  # [100, 256]
        intermediate_samples = data["intermediate_samples"]

        if 0 in intermediate_samples:
            inter_0 = dataset.inverse_scale(intermediate_samples[0])
            if not torch.allclose(sequences, inter_0, rtol=1e-5, atol=1e-8):
                print(f"Warning: Year {year}: sequences != intermediate_samples[0]")

        intermediate_samples = {round(t): samples for t, samples in intermediate_samples.items() if t <= 500}

        year_metrics_list = []
        year_gen_samples = []
        valid_indices = torch.logical_not(torch.isnan(sequences).any(dim=1) | torch.isinf(sequences).any(dim=1))
        valid_sequences = sequences[valid_indices]

        if len(valid_sequences) == 0:
            print(f"Warning: No valid sequences for year {year}")
            continue

        for i, gen_data in enumerate(valid_sequences):
            gen_metrics = calculate_metrics(gen_data.unsqueeze(0))
            year_metrics_list.append(gen_metrics)
            year_gen_samples.append(gen_data)

        metrics[f'yearly_{year}'] = average_metrics(year_metrics_list, store_individual=True)
        all_gen_samples.append(valid_sequences)

        for t in intermediate_samples:
            inter_samples = dataset.inverse_scale(intermediate_samples[t])[valid_indices]
            if torch.isnan(inter_samples).any() or torch.isinf(inter_samples).any():
                continue

            inter_metrics_list = [calculate_metrics(sample.unsqueeze(0)) for sample in inter_samples]
            inter_metrics = average_metrics(inter_metrics_list, store_individual=True)
            metrics_per_timestep['years'][year][t] = inter_metrics
            if t not in metrics_per_timestep['global']:
                metrics_per_timestep['global'][t] = []
            metrics_per_timestep['global'][t].extend(inter_metrics_list)

        save_visualizations(year_gen_samples, year, output_dir)
        with open(os.path.join(output_dir, f'metrics_{year}.json'), 'w') as f:
            json.dump(metrics[f'yearly_{year}'], f, indent=2)

    if all_gen_samples:
        gen_all = torch.cat(all_gen_samples, dim=0)
        global_metrics_list = [calculate_metrics(sample.unsqueeze(0)) for sample in gen_all
                               if not (torch.isnan(sample).any() or torch.isinf(sample).any())]
        metrics['global'] = average_metrics(global_metrics_list, store_individual=True)
        with open(os.path.join(output_dir, 'metrics_global.json'), 'w') as f:
            json.dump(metrics['global'], f, indent=2)

        for t in metrics_per_timestep['global']:
            metrics_per_timestep['global'][t] = average_metrics(metrics_per_timestep['global'][t], store_individual=True)

    print_enhanced_report(metrics, years)
    plot_metrics_vs_timesteps(metrics_per_timestep, output_dir, years, real_metrics_dir)
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    config = {
        "generated_dir": "generated_sequences/generation_20250605_140059",
        "output_dir": "validation_results/generated_20250605_140059",
        "data_path": "financial_data/sequences/sequences_256.pt",
        "real_metrics_dir": "real_metrics",
        "years": list(range(2017, 2024))
    }

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    validate_generated_data(config)