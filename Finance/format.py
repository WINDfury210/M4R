import torch
import os
from tqdm import tqdm

# Configuration
sequence_lengths = [63, 252]  # Hard-coded sequence lengths
input_file = "financial_data/sp500_returns.pt"
vix_file = "financial_data/vix.pt"
output_dir = "financial_data/sequences"
os.makedirs(output_dir, exist_ok=True)

# Load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data = torch.load(input_file, weights_only=False)
returns = data["returns"].to(device)  # [n_stocks, n_days], e.g., [98, 1508]
tickers = data["tickers"]
dates = data["dates"]
vix = torch.load(vix_file, weights_only=False).to(device)  # [n_days], e.g., [1508]

# Generate sequences for each length
for seq_len in sequence_lengths:
    sequences = []
    conditions = []
    print(f"Generating sequences of length {seq_len}...")
    step = seq_len // 4  # Dynamic step size, e.g., 15 for 63, 63 for 252
    for i in tqdm(range(returns.shape[0]), desc=f"Processing stocks (len={seq_len})"):
        stock_returns = returns[i]
        stock_vix = vix
        for start in range(0, len(stock_returns) - seq_len + 1, step):
            seq = stock_returns[start:start + seq_len]
            cond = stock_vix[start:start + seq_len].mean()
            if seq.shape[0] == seq_len and not torch.isnan(seq).any():
                sequences.append(seq)
                conditions.append(cond)
    
    # Stack sequences
    if sequences:
        sequences = torch.stack(sequences)
        conditions = torch.stack(conditions).unsqueeze(-1)
        output_file = os.path.join(output_dir, f"sequences_{seq_len}.pt")
        torch.save({"sequences": sequences.cpu(), "conditions": conditions.cpu(), "tickers": tickers}, 
                   output_file)
        print(f"Saved {sequences.shape[0]} sequences of length {seq_len} to {output_file}")
    else:
        print(f"No sequences generated for length {seq_len}")