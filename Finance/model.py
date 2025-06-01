"""
Diffusion Model Training Script
Train ConditionalUNet1D with MSE, ACF, Std, and Mean losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Model Definitions --------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim, embedding_type="sinusoidal", hidden_dim=1024):
        super().__init__()
        self.dim = dim
        self.embedding_type = embedding_type
        if embedding_type == "sinusoidal":
            half_dim = dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            self.register_buffer("emb", emb)
        elif embedding_type == "linear":
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            )

    def forward(self, time):
        if self.embedding_type == "sinusoidal":
            emb = time[:, None] * self.emb[None, :]
            emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
            if self.dim % 2 == 1:
                emb = F.pad(emb, (0, 1, 0, 0))
            return emb
        elif self.embedding_type == "linear":
            time = time.unsqueeze(-1).float()
            return self.mlp(time)

class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, ch, seq_len = x.size()
        q = self.query(x).view(batch, -1, seq_len).permute(0, 2, 1)  # [batch, seq_len, ch//8]
        k = self.key(x).view(batch, -1, seq_len)  # [batch, ch//8, seq_len]
        v = self.value(x).view(batch, -1, seq_len)  # [batch, ch, seq_len]
        attn = self.softmax(torch.bmm(q, k) / (ch // 8) ** 0.5)  # [batch, seq_len, seq_len]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch, ch, seq_len)  # [batch, ch, seq_len]
        return x + self.gamma * out

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = out.permute(0, 2, 1)  # [batch_size, seq_len, out_channels]
        out = self.ln1(out)
        out = out.permute(0, 2, 1)  # [batch_size, out_channels, seq_len]
        out = self.relu(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = out.permute(0, 2, 1)
        out += residual
        return self.relu(out)

class ConditionalUNet1D(nn.Module):
    def __init__(self, seq_len=256, channels=[32, 64, 128, 256]):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_levels = len(channels)
        
        self.time_embed = TimeEmbedding(dim=channels[-1], embedding_type="sinusoidal")
        self.date_embed = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, channels[-1]),
            nn.LayerNorm(channels[-1])
        )
        self.input_conv = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)
        self.encoder_convs = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        self.attentions = nn.ModuleList()
        in_channels = channels[0]
        for i, out_channels in enumerate(channels):
            self.encoder_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                                              stride=2 if i>0 else 1, padding=1))
            self.encoder_res.append(ResidualBlock1D(out_channels, out_channels))
            self.attentions.append(SelfAttention1D(out_channels) if 0<i<len(channels) else nn.Identity())
            in_channels = out_channels
        self.mid_block1 = ResidualBlock1D(channels[-1], channels[-1])
        self.mid_block2 = ResidualBlock1D(channels[-1], channels[-1])
        self.decoder_convs = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        for i in range(len(channels)-1):
            in_channels = channels[-1-i] + channels[-1-i]
            out_channels = channels[-2-i]
            self.decoder_convs.append(nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ))
            self.decoder_res.append(ResidualBlock1D(out_channels, out_channels))
        self.final_conv = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x, t, date):
        time_emb = self.time_embed(t)
        date_emb = self.date_embed(date)
        combined_cond = time_emb + date_emb
        x = x.unsqueeze(1)
        x = self.input_conv(x)
        skips = []
        for i, (conv, res, attn) in enumerate(zip(self.encoder_convs, self.encoder_res, self.attentions)):
            x = F.relu(conv(x))
            x = res(x)
            x = attn(x)
            skips.append(x)
        x = self.mid_block1(x)
        cond = combined_cond.unsqueeze(-1)
        x = x + cond
        x = self.mid_block2(x)
        for i, (conv, res) in enumerate(zip(self.decoder_convs, self.decoder_res)):
            skip = skips[-(i+1)]
            if x.shape[-1] != skip.shape[-1]:
                if x.shape[-1] < skip.shape[-1]:
                    pad_len = skip.shape[-1] - x.shape[-1]
                    x = F.pad(x, (0, pad_len))
                else:
                    x = x[:, :, :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = F.relu(conv(x))
            x = res(x)
        x = self.final_conv(x).squeeze(1)
        return x