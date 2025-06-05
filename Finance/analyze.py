import os
import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf

# 1. Metrics Calculation -----------------------------------------------------

def calculate_metrics(data, dummy=None):
    metrics = {}
    
    if data.dim() == 1:
        data = data.unsqueeze(0)
    
    # Check for NaN/Inf or all zeros
    if torch.isnan(data).any() or torch.isinf(data).any() or data.numel() == 0 or torch.all(data == 0):
        print(f"Warning: Invalid data in calculate_metrics (shape: {data.shape}, NaN: {torch.isnan(data).any()}, Inf: {torch.isinf(data).any()}, all zeros: {torch.all(data == 0)})")
        return None
    
    data_np = data.cpu().numpy()
    if data_np.ndim == 1:
        data_np = data_np[np.newaxis, :]
    
    metrics['gen_mean'] = float(np.mean(data_np, axis=1)[0]) if not np.isnan(data_np.mean()) else 0.0
    metrics['gen_std'] = float(np.std(data_np, axis=1)[0]) if data_np.size > 1 and not np.isnan(data_np.std()) else 0.0
    
    sample = data_np[0]
    if len(sample) > 1 and np.var(sample) > 1e-10 and not np.isnan(sample).any():
        lagged = sample[:-1]
        next_val = sample[1:]
        metrics['gen_corr'] = np.corrcoef(lagged, next_val)[0, 1] if len(lagged) > 1 else 0.0
        try:
            gen_acf = acf(sample, nlags=20, fft=True)[1:]
            metrics['gen_acf'] = float(gen_acf.mean()) if not np.isnan(gen_acf).all() else 0.0
        except Exception as e:
            print(f"Warning: ACF computation failed: {e}")
            metrics['gen_acf'] = 0.0
    else:
        metrics['gen_corr'] = 0.0
        metrics['gen_acf'] = 0.0
    
    metrics['gen_skew'] = float(stats.skew(sample)) if len(sample) > 2 and not np.isnan(sample).any() else 0.0
    metrics['gen_kurt'] = float(stats.kurtosis(sample)) if len(sample) > 3 and not np.isnan(sample).any() else 0.0
    
    abs_data_np = np.abs(data_np)
    metrics['abs_gen_mean'] = float(np.mean(abs_data_np, axis=1)[0]) if not np.isnan(abs_data_np.mean()) else 0.0
    metrics['abs_gen_std'] = float(np.std(abs_data_np, axis=1)[0]) if abs_data_np.size > 1 and not np.isnan(abs_data_np.std()) else 0.0
    
    abs_sample = abs_data_np[0]
    if len(abs_sample) > 1 and np.var(abs_sample) > 1e-10 and not np.isnan(abs_sample).any():
        abs_lagged = abs_sample[:-1]
        abs_next = abs_sample[1:]
        metrics['abs_gen_corr'] = np.corrcoef(abs_lagged, abs_next)[0, 1] if len(lagged) > 1 else 0.0
        try:
            abs_gen_acf = acf(abs_sample, nlags=20, fft=True)[1:]
            metrics['abs_gen_acf'] = float(abs_gen_acf.mean()) if not np.isnan(abs_gen_acf).all() else 0.0
        except Exception as e:
            print(f"Warning: Abs ACF computation failed: {e}")
            metrics['abs_gen_acf'] = 0.0
    else:
        metrics['abs_gen_corr'] = 0.0
        metrics['abs_gen_acf'] = 0.0
    
    metrics['abs_gen_skew'] = float(stats.skew(abs_sample)) if len(abs_sample) > 2 and not np.isnan(abs_sample).any() else 0.0
    metrics['abs_gen_kurt'] = float(stats.kurtosis(abs_sample)) if len(abs_sample) > 3 and not np.isnan(abs_sample).any() else 0.0
    
    return metrics

def average_metrics(metrics_list, store_individual=False):
    if not metrics_list:
        return {
            'gen_mean': {'mean': 0.0, 'variance': 0.0},
            'gen_std': {'mean': 0.0, 'variance': 0.0},
            'gen_corr': {'mean': 0.0, 'variance': 0.0},
            'gen_acf': {'mean': 0.0, 'variance': 0.0},
            'gen_skew': {'mean': 0.0, 'variance': 0.0},
            'gen_kurt': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_mean': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_std': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_corr': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_acf': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_skew': {'mean': 0.0, 'variance': 0.0},
            'abs_gen_kurt': {'mean': 0.0, 'variance': 0.0}
        }
    
    stats = {}
    keys = list(metrics_list[0].keys())
    for key in keys:
        values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        if values:
            stats[key] = {
                'mean': float(np.mean(values)),
                'variance': float(np.var(values)) if len(values) > 1 else 0.0
            }
            if store_individual:
                stats[key]['means'] = values
        else:
            stats[key] = {
                'mean': 0.0,
                'variance': 0.0
            }
            if store_individual:
                stats[key]['means'] = []
    return stats

# 2. Visualization Functions -------------------------------------------------

def save_visualizations(gen_samples, year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if not gen_samples:
        print(f"Warning: No generated samples for year {year}, skipping visualization")
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
    try:
        gen_acf = acf(gen_sample, nlags=20, fft=True)
    except:
        gen_acf = np.zeros(21)
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
    try:
        abs_gen_acf = acf(abs_gen_sample, nlags=20, fft=True)
    except:
        abs_gen_acf = np.zeros(21)
    plt.stem(range(len(abs_gen_acf)), abs_gen_acf, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title(f"Abs Autocovariance (Year {year})")
    plt.xlabel("Lag")
    plt.ylabel("Autocovariance")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'year_{year}_absolute_sample.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 6))
    stats.probplot(gen_sample, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot (Year {year})")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
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
    
    # Global plot
    global_timesteps = sorted(metrics_per_timestep['global'].keys())
    if global_timesteps:
        try:
            with open(os.path.join(real_metrics_dir, 'real_metrics_global.json'), 'r') as f:
                real_global = json.load(f)
        except FileNotFoundError:
            print(f"Warning: real_metrics_global.json not found in {real_metrics_dir}")
            real_global = {}
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics_to_plot, 1):
            means = [metrics_per_timestep['global'][t].get(metric, {}).get('mean', 0.0) for t in global_timesteps]
            quantiles = [
                np.percentile(metrics_per_timestep['global'][t].get(metric, {}).get('means', [0.0]), [25, 75])
                if metrics_per_timestep['global'][t].get(metric, {}).get('means', []) else [0.0, 0.0]
                for t in global_timesteps
            ]
            q1 = [q[0] for q in quantiles]
            q3 = [q[1] for q in quantiles]
            
            plt.subplot(2, 3, i)
            plt.plot(global_timesteps[::-1], means, color='blue', label='Generated Mean')
            plt.fill_between(global_timesteps[::-1], q1, q3, color='blue', alpha=0.2, label='25%-75% Quantile')
            
            real_metric = real_metrics_map[metric]
            real_value = real_global.get(real_metric, {}).get('mean', None)
            if real_value is not None:
                plt.axhline(real_value, color='red', linestyle='--', label='Real Mean')
            
            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Timestep')
            plt.ylabel('Metric Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_vs_timesteps_global.png'), dpi=300)
        plt.close()
    
    # Yearly plots
    for year in years:
        if year not in metrics_per_timestep['years']:
            print(f"Warning: No metrics for year {year}")
            continue
        
        year_timesteps = sorted(metrics_per_timestep['years'][year].keys())
        if not year_timesteps:
            print(f"Warning: No timesteps for year {year}")
            continue
        
        try:
            with open(os.path.join(real_metrics_dir, f'real_metrics_{year}.json'), 'r') as f:
                real_year = json.load(f)
        except FileNotFoundError:
            print(f"Warning: real_metrics_{year}.json not found in {real_metrics_dir}")
            real_year = {}
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics_to_plot, 1):
            means = [metrics_per_timestep['years'][year][t].get(metric, {}).get('mean', 0.0) for t in year_timesteps]
            quantiles = [
                np.percentile(metrics_per_timestep['years'][year][t].get(metric, {}).get('means', [0.0]), [25, 75])
                if metrics_per_timestep['years'][year][t].get(metric, {}).get('means', []) else [0.0, 0.0]
                for t in year_timesteps
            ]
            q1 = [q[0] for q in quantiles]
            q3 = [q[1] for q in quantiles]
            
            plt.subplot(2, 3, i)
            plt.plot(year_timesteps[::-1], means, color='blue', label='Generated Mean')
            plt.fill_between(year_timesteps[::-1], q1, q3, color='blue', alpha=0.2, label='25%-75% Quantile')
            
            real_metric = real_metrics_map[metric]
            real_value = real_year.get(real_metric, {}).get('mean', None)
            if real_value is not None:
                plt.axhline(real_value, color='red', linestyle='--', label='Real Mean')
            
            plt.title(f"{metric.replace('_', ' ').title()} (Year {year})")
            plt.xlabel('Timestep')
            plt.ylabel('Metric Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_{year}.png'), dpi=300)
        plt.close()

# 3. Report Generation -------------------------------------------------------

def print_enhanced_report(metrics_dict, years):
    print("\n=== Validation Report ===")
    
    print("\n[Global Statistics]")
    print(f"{'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 39)
    global_stats = metrics_dict.get('global', {})
    
    if all(isinstance(global_stats.get(k), dict) for k in global_stats):
        for metric in ['gen_mean', 'gen_std', 'gen_corr', 'gen_acf', 'gen_skew', 'gen_kurt']:
            mean = global_stats.get(metric, {}).get('mean', 0.0)
            variance = global_stats.get(metric, {}).get('variance', 0.0)
            print(f"{metric:<15} {mean:>12.6f} {variance:>12.6f}")
    else:
        for metric in ['gen_mean', 'gen_std', 'gen_corr', 'gen_acf', 'gen_skew', 'gen_kurt']:
            mean = global_stats.get(metric, 0.0)
            variance = 0.0
            print(f"{metric:<15} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Absolute Value Global Statistics]")
    print(f"{'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 39)
    if all(isinstance(global_stats.get(k), dict) for k in global_stats):
        for metric in ['abs_gen_mean', 'abs_gen_std', 'abs_gen_corr', 'abs_gen_acf', 'abs_gen_skew', 'abs_gen_kurt']:
            mean = global_stats.get(metric, {}).get('mean', 0.0)
            variance = global_stats.get(metric, {}).get('variance', 0.0)
            print(f"{metric:<15} {mean:>12.6f} {variance:>12.6f}")
    else:
        for metric in ['abs_gen_mean', 'abs_gen_std', 'abs_gen_corr', 'abs_gen_acf', 'abs_gen_skew', 'abs_gen_kurt']:
            mean = global_stats.get(metric, 0.0)
            variance = 0.0
            print(f"{metric:<15} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Yearly Statistics]")
    print(f"{'Year':<6} {'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 45)
    for year in years:
        year_stats = metrics_dict.get(f'gen_metrics_{year}', {})
        for metric in ['gen_mean', 'gen_std', 'gen_corr', 'gen_acf', 'gen_skew', 'gen_kurt']:
            mean = year_stats.get(metric, {}).get('mean', 0.0)
            variance = year_stats.get(metric, {}).get('variance', 0.0)
            print(f"{year:<6} {metric:<15} {mean:>12.6f} {variance:>12.6f}")
    
    print("\n[Absolute Value Yearly Statistics]")
    print(f"{'Year':<6} {'Metric':<15} {'Mean':>12} {'Variance':>12}")
    print("-" * 45)
    for year in years:
        year_stats = metrics_dict.get(f'gen_metrics_{year}', {})
        for metric in ['abs_gen_mean', 'abs_gen_std', 'abs_gen_corr', 'abs_gen_acf', 'abs_gen_skew', 'abs_gen_kurt']:
            mean = year_stats.get(metric, {}).get('mean', 0.0)
            variance = year_stats.get(metric, {}).get('variance', 0.0)
            print(f"{year:<6} {metric:<15} {mean:>12.6f} {variance:>12.6f}")

# 4. Main Validation Function -----------------------------------------------

def validate_generated_data(config):
    years = config.get("years", list(range(2017, 2024)))
    generated_dir = config["generated_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
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
        sequences = data["sequences"]  # Shape: [100, 256]
        intermediate_samples = data["intermediate_samples"]  # {t: [100, 256]}
        
        
        # Map timesteps (identity mapping)
        intermediate_samples_new = {}
        for t in intermediate_samples:
            if t > 1000:
                continue
            t_new = round(t)
            intermediate_samples_new[t_new] = intermediate_samples[t]
        intermediate_samples = intermediate_samples_new
        
        
        year_metrics_list = []
        year_gen_samples = []
        
        # Process final sequences
        for i in range(len(sequences)):
            gen_data = sequences[i:i+1]  # Shape: [1, 256]
            if torch.isnan(gen_data).any() or torch.isinf(gen_data).any():
                print(f"Warning: Invalid data in sample {i} (NaN: {torch.isnan(gen_data).any()}, Inf: {torch.isinf(gen_data).any()})")
                continue
            try:
                mean_val = gen_data.mean().item()
                std_val = gen_data.std().item()
                print(f"Year {year}, Sample {i}: mean={mean_val:.6f}, std={std_val:.6f}")
            except ValueError as e:
                print(f"Error formatting sample {i}: mean={gen_data.mean().item()}, std={gen_data.std().item()}, error={e}")
                continue
            gen_metrics = calculate_metrics(gen_data)

            if gen_metrics is None:
                print(f"Skipping sample {i} due to invalid metrics")
                continue
            year_metrics_list.append(gen_metrics)
            year_gen_samples.append(gen_data.squeeze())

            year_metrics_list.append(gen_metrics)
            year_gen_samples.append(gen_data.squeeze())
        
        metrics[f'gen_metrics_{year}'] = average_metrics(year_metrics_list, store_individual=True)
        all_gen_samples.append(sequences)
        
        # Process intermediate samples
        for t in intermediate_samples:
            inter_samples = intermediate_samples[t]  # [100, 256]
            # Inverse scale to original scale
            
            if torch.isnan(inter_samples).any() or torch.isinf(inter_samples).any():
                print(f"Warning: Invalid data in timestep {t} (NaN: {torch.isnan(inter_samples).any()}, Inf: {torch.isinf(inter_samples).any()})")
                continue
            try:
                print(f"Year {year}, Timestep {t}: mean={inter_samples.mean().item():.6f}, std={inter_samples.std().item():.6f}")
            except ValueError as e:
                print(f"Error formatting timestep {t}: mean={inter_samples.mean().item()}, std={inter_samples.std().item()}, error={e}")
                continue
            
            # Compute metrics for each sample
            inter_metrics_list = []
            for i in range(len(inter_samples)):
                sample = inter_samples[i:i+1]  # [1, 256]
                inter_metrics = calculate_metrics(sample)
                if inter_metrics is None:
                    print(f"Skipping intermediate sample {i} at timestep {t}")
                    continue
                inter_metrics_list.append(inter_metrics)
            
            # Store individual metrics with 'means'
            inter_metrics = average_metrics(inter_metrics_list, store_individual=True)
            metrics_per_timestep['years'][year][t] = inter_metrics
            if t not in metrics_per_timestep['global']:
                metrics_per_timestep['global'][t] = []
            metrics_per_timestep['global'][t].extend(inter_metrics_list)
        
        save_visualizations(year_gen_samples, year, output_dir)
        with open(os.path.join(output_dir, f'metrics_{year}.json'), 'w') as f:
            json.dump(metrics[f'gen_metrics_{year}'], f, indent=2)
    
    # Global metrics
    if all_gen_samples:
        gen_all = torch.cat(all_gen_samples, dim=0)  # Shape: [700, 256]
        try:
            print(f"Global: shape={gen_all.shape}, mean={gen_all.mean().item():.6f}, std={gen_all.std().item():.6f}")
        except ValueError as e:
            print(f"Error formatting global: mean={gen_all.mean().item()}, std={gen_all.std().item()}, error={e}")
        
        # Compute metrics for each sample
        global_metrics_list = []
        for i in range(gen_all.shape[0]):
            sample = gen_all[i:i+1]  # Shape: [1, 256]
            if torch.isnan(sample).any() or torch.isinf(sample).any():
                print(f"Warning: Invalid global sample {i} (NaN: {torch.isnan(sample).any()}, Inf: {torch.isinf(sample).any()})")
                continue
            global_metrics = calculate_metrics(sample)
            
            global_metrics_list.append(global_metrics)
        

        metrics['global'] = average_metrics(global_metrics_list, store_individual=True)
        with open(os.path.join(output_dir, 'metrics_global.json'), 'w') as f:
            json.dump(metrics['global'], f, indent=2)
        
        # Global intermediate samples
        for t in metrics_per_timestep['global']:
            global_metrics_list = metrics_per_timestep['global'][t]
            metrics_per_timestep['global'][t] = average_metrics(global_metrics_list, store_individual=True)
    else:
        print("Warning: No global samples found")
        metrics['global'] = average_metrics([])
    
    
    print_enhanced_report(metrics, years)
    plot_metrics_vs_timesteps(metrics_per_timestep, output_dir, years, config.get("real_metrics_dir", "real_metrics"))
    
    print(f"\nMetrics and visualizations saved to {output_dir}")
    print(f"Validation complete!")

if __name__ == "__main__":
    config = {
        "generated_dir": "generated_sequences/generation_20250605_022420",
        "output_dir": "validation_results/generated_20250605_022420",
        "data_path": "financial_data/sequences/sequences_256.pt",
        "real_metrics_dir": "real_metrics",
        "years": list(range(2017, 2024))
    }
    
    # Fix random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(0)
    
    validate_generated_data(config)