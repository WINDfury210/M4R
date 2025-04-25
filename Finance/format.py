import torch
import os
from tqdm import tqdm

# Configuration
data_dir = "financial_data"
output_dir = os.path.join(data_dir, "sequences")
os.makedirs(output_dir, exist_ok=True)
sequence_lengths = [63, 252]
stride = 63

# Load data
sp500_data = torch.load(os.path.join(data_dir, "sp500_returns.pt"), weights_only=False)
vix_data = torch.load(os.path.join(data_dir, "vix.pt"), weights_only=False)

# Format sequences
for seq_len in sequence_lengths:
    sequences = []
    conditions = []
    n_stocks, n_days = sp500_data.shape
    for stock_idx in tqdm(range(n_stocks), desc=f"Formatting sequences (length={seq_len})"):
        stock_data = sp500_data[stock_idx]
        for start in range(0, n_days - seq_len + 1, stride):
            sequence = stock_data[start:start + seq_len]
            vix_sequence = vix_data[start:start + seq_len]  # VIX sequence
            if sequence.shape[0] == seq_len and vix_sequence.shape[0] == seq_len:
                sequences.append(sequence)
                conditions.append(vix_sequence)
    
    # Save sequences
    sequences = torch.stack(sequences)  # [n_sequences, seq_len]
    conditions = torch.stack(conditions)  # [n_sequences, seq_len]
    output_file = os.path.join(output_dir, f"sequences_{seq_len}.pt")
    torch.save({"sequences": sequences, "conditions": conditions}, output_file)
    print(f"Saved {sequences.shape[0]} sequences of length {seq_len} to {output_file}")