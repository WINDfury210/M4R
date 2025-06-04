import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
from glob import glob
from scipy import stats
from model import ConditionalUNet1D
from tqdm import tqdm
import time

# 2. Diffusion Process -------------------------------------------------------

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

# 3. Data Loading ------------------------------------------------------------

class FinancialDataset(Dataset):
    def __init__(self, data_path, scale_factor=1.0, preload=True):
        data = torch.load(data_path, map_location='cpu')
        self.sequences = data["sequences"]
        self.dates = data["start_dates"]
        if preload:
            self.sequences = self.sequences.to('cpu', non_blocking=True)
            self.dates = self.dates.to('cpu', non_blocking=True)
        self.original_mean = self.sequences.mean().item()
        self.original_std = self.sequences.std().item()
        self.scale_factor = scale_factor
        self.sequences = (self.sequences - self.original_mean) / self.original_std * scale_factor
        print(f"Scaled data - Mean: {self.sequences.mean().item():.6f}, Std: {self.sequences.std().item():.6f}")
        
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
            norm_year = (year - min_year) / (max_year - min_year)
            start_date = torch.tensor([norm_year, 0.0, 0.0], dtype=torch.float32)
            start_dates.append(start_date)
        return torch.stack(start_dates)
    
    def get_real_sequences_for_year(self, annual_date, num_samples=10):
        date_diffs = torch.norm(self.dates - annual_date, dim=1)
        closest_indices = torch.argsort(date_diffs)[:num_samples]
        return self.sequences[closest_indices], closest_indices

# 4. Loss Functions -----------------------------------------------------------

def acf_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    pred_acf = torch.fft.irfft(pred_fft * pred_fft.conj(), dim=-1)
    target_acf = torch.fft.irfft(target_fft * target_fft.conj(), dim=-1)
    pred_acf = pred_acf[:, 1:21] / (pred_acf[:, 0:1] + 1e-8)
    target_acf = target_acf[:, 1:21] / (target_acf[:, 0:1] + 1e-8)
    return F.mse_loss(pred_acf, target_acf)

def std_loss(pred, target):
    pred_std = pred.std(dim=-1)
    target_std = target.std(dim=-1)
    return F.mse_loss(pred_std, target_std)

def mean_loss(pred, target):
    pred_mean = pred.mean(dim=-1)
    target_mean = target.mean(dim=-1)
    return F.mse_loss(pred_mean, target_mean)

def ks_loss(pred, target):
    batch_size = pred.size(0)
    losses = []
    for i in range(batch_size):
        p = pred[i].detach().cpu().numpy()
        t = target[i].detach().cpu().numpy()
        ks_stat, _ = stats.ks_2samp(p, t)
        losses.append(torch.tensor(ks_stat, device=pred.device))
    return torch.stack(losses).mean()

# 5. Training Function --------------------------------------------------------

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'None'}")
    
    model = ConditionalUNet1D(seq_len=256, channels=config["channels"]).to(device)
    diffusion = DiffusionProcess(num_timesteps=config["num_timesteps"], device=device)
    dataset = FinancialDataset(config["data_path"], scale_factor=1.0, preload=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")
    
    # Load checkpoint
    start_epoch = 0
    checkpoint_files = sorted(
        glob(os.path.join(config["save_dir"], "model2_epoch_*.pth")),
        key=lambda x: int(re.search(r'model2_epoch_(\d+).pth', x).group(1))
    )
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from checkpoint: {latest_checkpoint}, starting at epoch {start_epoch}")
        except Exception as e:
            print(f"Checkpoint loading failed: {e}, starting from scratch")
    
    # Manage checkpoints to save space
    max_checkpoints = 2
    def clean_old_checkpoints():
        checkpoints = sorted(
            glob(os.path.join(config["save_dir"], "model2_epoch_*.pth")),
            key=lambda x: int(re.search(r'model2_epoch_(\d+).pth', x).group(1))
        )
        if len(checkpoints) > max_checkpoints:
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
    
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config['num_epochs']}, Loss: 0.000000",
            total=num_batches,
            leave=True,
            mininterval=1.0
        )
        
        for batch in progress_bar:
            sequences = batch["sequence"].to(device, non_blocking=True)
            dates = batch["date"].to(device, non_blocking=True)
            
            t = torch.randint(0, diffusion.num_timesteps, (sequences.size(0),), device=device)
            noisy_x, noise = diffusion.add_noise(sequences, t)
            
            optimizer.zero_grad(set_to_none=True)
            
            try:
                with torch.amp.autocast('cuda',enabled=device.type == "cuda"):
                    pred_noise = model(noisy_x, t, dates)
                    mse_loss = F.mse_loss(pred_noise, noise)
                    # acf_loss_val = acf_loss(pred_noise, noise)
                    # std_loss_val = std_loss(pred_noise, noise)
                    # mean_loss_val = mean_loss(pred_noise, noise)
                    # ks_loss_val = ks_loss(pred_noise, noise)
                    loss = mse_loss  # + 0.1 * std_loss_val  # Uncomment to enable std_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                avg_loss = total_loss / (progress_bar.n + 1)
                progress_bar.set_description(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.6f}")
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch_size={config['batch_size']}, reducing to {config['batch_size']//2}")
                    config["batch_size"] //= 2
                    if config["batch_size"] < 8:
                        raise RuntimeError("Batch size too small, stopping training")
                    torch.cuda.empty_cache()
                    return train_model(config)  # Retry with smaller batch_size
                else:
                    raise e
        
        progress_bar.close()
        scheduler.step()
        
        if (epoch + 1) % config["save_interval"] == 0 or epoch == config["num_epochs"] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config["save_dir"], f"model2_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoint: {os.path.join(config['save_dir'], f'model2_epoch_{epoch+1}.pth')}")
            clean_old_checkpoints()
    
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "final_model.pth"))
    print(f"Saved final model: {os.path.join(config['save_dir'], 'final_model.pth')}")

# 6. Main --------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "data_path": "financial_data/sequences/sequences_256.pt",
        "save_dir": "saved_models",
        "num_epochs": 4000,
        "batch_size": 128,  # Reduced from 64 to avoid OOM
        "num_timesteps": 1000,
        "channels": [32, 128, 512, 2048],
        "lr": 1e-4,
        "save_interval": 50,  # More frequent for network interruptions
        "num_workers": 8  # Increased for faster data loading
    }
    os.makedirs(config["save_dir"], exist_ok=True)
    train_model(config)