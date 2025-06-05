import os
import json
import numpy as np
import torch
from scipy import stats
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

class SequenceAnalyzer:
    def __init__(self, real_metrics_dir="real_metrics"):
        self.real_metrics_dir = real_metrics_dir
    
    def load_generated_data(self, generation_dir):
        data_files = [f for f in os.listdir(generation_dir) 
                     if f.startswith("generated_") and f.endswith(".pt")]
        all_data = {}
        
        for filename in data_files:
            year = int(filename.split("_")[1].split(".")[0])
            filepath = os.path.join(generation_dir, filename)
            year_data = torch.load(filepath)
            all_data[year] = {
                "sequences": year_data["sequences"],
                "metadata": year_data["metadata"]
            }
        
        return all_data
    
    def load_real_metrics(self, year):
        metrics_path = os.path.join(self.real_metrics_dir, f"real_metrics_{year}.json")
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def calculate_metrics(self, sequences):
        """Calculate metrics for a batch of sequences (num_samples, seq_len)"""
        metrics = {
            "gen_mean": np.mean(sequences),
            "gen_std": np.std(sequences),
            "gen_skew": stats.skew(sequences, axis=1).mean(),
            "gen_kurt": stats.kurtosis(sequences, axis=1).mean(),
            "gen_corr": self._calculate_autocorr(sequences),
            "gen_acf": self._calculate_acf(sequences),
            
            "abs_gen_mean": np.mean(np.abs(sequences)),
            "abs_gen_std": np.std(np.abs(sequences)),
            "abs_gen_skew": stats.skew(np.abs(sequences), axis=1).mean(),
            "abs_gen_kurt": stats.kurtosis(np.abs(sequences), axis=1).mean(),
            "abs_gen_corr": self._calculate_autocorr(np.abs(sequences)),
            "abs_gen_acf": self._calculate_acf(np.abs(sequences))
        }
        return metrics
    
    def _calculate_autocorr(self, sequences):
        """Calculate average autocorrelation"""
        corrs = []
        for seq in sequences:
            if len(seq) > 1:
                corr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
                corrs.append(corr)
        return np.mean(corrs) if corrs else 0.0
    
    def _calculate_acf(self, sequences, nlags=20):  # 修改为nlags=20以保持一致性
        """Calculate average ACF"""
        acfs = []
        for seq in sequences:
            try:
                acf_vals = acf(seq, nlags=nlags, fft=True)[1:]
                acfs.append(np.mean(acf_vals))
            except:
                acfs.append(0.0)
        return np.mean(acfs)
    
    def compare_with_real(self, gen_metrics, real_metrics):
        """Compare generated metrics with real data metrics"""
        comparison = {}
        for key in gen_metrics:
            if key.startswith("gen_"):
                real_key = key.replace("gen_", "real_")
                comparison[key] = {
                    "generated": gen_metrics[key],
                    "real": real_metrics.get(real_key, {}).get("mean", 0),
                    "diff": gen_metrics[key] - real_metrics.get(real_key, {}).get("mean", 0)
                }
        return comparison
    
    def plot_metrics_vs_timesteps(self, metrics_per_timestep, real_metrics, year, output_dir):
        """
        按照您的要求修改的绘图函数：
        1. 保持原始的时间步曲线和标准差区域
        2. 添加红色水平线表示真实值
        3. 完全保留原有可视化风格
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timesteps = sorted(metrics_per_timestep.keys(), reverse=False)  # From 1000 to 0
        if not timesteps:
            print(f"Warning: No metrics available for timesteps (Year {year})")
            return
        
        # 配置要绘制的指标及对应的真实数据键名
        metric_config = [
            {'gen_key': 'gen_mean', 'real_key': 'real_mean', 'title': 'Mean'},
            {'gen_key': 'gen_std', 'real_key': 'real_std', 'title': 'Std Dev'},
            {'gen_key': 'gen_kurt', 'real_key': 'real_kurt', 'title': 'Kurtosis'},
            {'gen_key': 'abs_gen_mean', 'real_key': 'abs_real_mean', 'title': 'Abs Mean'},
            {'gen_key': 'abs_gen_std', 'real_key': 'abs_real_std', 'title': 'Abs Std'},
            {'gen_key': 'abs_gen_kurt', 'real_key': 'abs_real_kurt', 'title': 'Abs Kurtosis'}
        ]
        
        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Metrics vs Timesteps (Year {year})', y=1.02)
        
        for i, config in enumerate(metric_config, 1):
            # 获取生成数据的指标
            means = [metrics_per_timestep[t].get(config['gen_key'], {}).get('mean', 0.0) 
                    for t in timesteps]
            variances = [metrics_per_timestep[t].get(config['gen_key'], {}).get('variance', 0.0) 
                        for t in timesteps]
            
            # 获取真实数据指标
            real_value = real_metrics.get(config['real_key'], {}).get('mean', 0.0)
            
            plt.subplot(2, 3, i)
            
            # 保持原有绘图逻辑
            plt.plot(timesteps, means, 'b-', label='Generated')
            plt.fill_between(timesteps,
                            [m - np.sqrt(v) for m, v in zip(means, variances)],
                            [m + np.sqrt(v) for m, v in zip(means, variances)],
                            color='b', alpha=0.2, label='±1 Std')
            
            # 添加红色水平线表示真实值
            plt.axhline(y=real_value, color='r', linestyle='--', linewidth=1.5, label='Real')
            
            plt.title(config['title'])
            plt.xlabel('Timestep')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            
            # 只在第一个子图显示图例
            if i == 1:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'metrics_vs_timesteps_{year}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def analyze_timestep_metrics(self, intermediate_samples, real_metrics, year, output_dir):
        """
        新增方法：处理中间时间步指标并绘图
        """
        metrics_per_timestep = {}
        for t, samples in intermediate_samples.items():
            samples_np = samples.numpy() if torch.is_tensor(samples) else samples
            metrics_per_timestep[t] = self.calculate_metrics(samples_np)
        
        return self.plot_metrics_vs_timesteps(metrics_per_timestep, real_metrics, year, output_dir)

def main():
    # Configuration
    generation_dir = "generated_sequences/generation_20230601_143000"  # Example path
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SequenceAnalyzer()
    
    # Load generated data
    print("Loading generated data...")
    generated_data = analyzer.load_generated_data(generation_dir)
    
    # Analyze each year
    results = {}
    for year, data in generated_data.items():
        print(f"Analyzing year {year}...")
        
        # Load real metrics
        real_metrics = analyzer.load_real_metrics(year)
        
        # Calculate metrics for generated sequences
        sequences = data["sequences"].numpy()
        gen_metrics = analyzer.calculate_metrics(sequences)
        
        # Compare with real data
        comparison = analyzer.compare_with_real(gen_metrics, real_metrics)
        
        # Save results
        results[year] = {
            "generated_metrics": gen_metrics,
            "real_metrics": real_metrics,
            "comparison": comparison,
            "metadata": data["metadata"]
        }
        
        # 如果有中间时间步数据，绘制时间步变化图
        if "intermediate_samples" in data:
            analyzer.analyze_timestep_metrics(
                data["intermediate_samples"],
                real_metrics,
                year,
                output_dir
            )
    
    # Save comprehensive results
    results_path = os.path.join(output_dir, "analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()