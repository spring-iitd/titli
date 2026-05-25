import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

try:
    from datasets import PcapWindowedDataset_lipschitz as PcapDataset
except ImportError:
    print("Warning: Could not import PcapDataset. Make sure datasets.py is in the same directory.")
    PcapDataset = None

try:
    from datasets import PcapTimeWindowedDataset_lipschitz, time_windowed_collate
except ImportError:
    PcapTimeWindowedDataset_lipschitz = None
    time_windowed_collate = None


@dataclass
class ModelConfig:
    model_type: str = ""
    input_size: int = 144
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 115
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    min_delta: float = 0.001
    activation: str = 'none'
    seq_len: int = 5
    weight_decay: float = 1e-3
    # Transformer-specific (unused for RNN-family)
    nhead: int = 4
    dim_feedforward: int = 256
    t_max: float = 10.0
    n_max: int = 512
    warmup_epochs: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def from_dict(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='none'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x.float(), h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self._apply_activation(out)

    def _apply_activation(self, out):
        if self.activation == 'relu':
            return torch.relu(out)
        elif self.activation == 'softplus':
            return torch.nn.functional.softplus(out)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'exp':
            return torch.exp(out)
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='none'):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x.float(), (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self._apply_activation(out)

    def _apply_activation(self, out):
        if self.activation == 'relu':
            return torch.relu(out)
        elif self.activation == 'softplus':
            return torch.nn.functional.softplus(out)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'exp':
            return torch.exp(out)
        return out


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='none'):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.bilstm(x.float(), (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self._apply_activation(out)

    def _apply_activation(self, out):
        if self.activation == 'relu':
            return torch.relu(out)
        elif self.activation == 'softplus':
            return torch.nn.functional.softplus(out)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'exp':
            return torch.exp(out)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='none'):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x.float(), h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self._apply_activation(out)

    def _apply_activation(self, out):
        if self.activation == 'relu':
            return torch.relu(out)
        elif self.activation == 'softplus':
            return torch.nn.functional.softplus(out)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'exp':
            return torch.exp(out)
        return out


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='none'):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.bigru(x.float(), h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self._apply_activation(out)

    def _apply_activation(self, out):
        if self.activation == 'relu':
            return torch.relu(out)
        elif self.activation == 'softplus':
            return torch.nn.functional.softplus(out)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'exp':
            return torch.exp(out)
        return out


class TransformerRoFex(nn.Module):
    """
    Variable-length Transformer feature extractor.

    Forward signature differs from the RNN family — accepts an explicit
    timestamp tensor and a key-padding mask:

        forward(x, timestamps, key_padding_mask=None) -> (B, output_size)

    Mechanisms:
      - Linear input projection to d_model
      - Sinusoidal positional encoding on cumulative inter-arrival times
      - Learnable [CLS] token; output is a linear head on the CLS embedding
      - Per-head learnable decay rate λ_h applied as additive attention bias
        B[b,h,i,j] = -softplus(λ_h) * (t_max_b - t_j)
        — older keys get more negative bias → less attention weight, mirroring
        AfterImage's exponential time-decay over multiple time scales.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 activation='none', nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        if hidden_size % nhead != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by nhead ({nhead})")
        self.d_model = hidden_size
        self.nhead = nhead
        self.activation = activation

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)
        # Clip raw IAT field (input index 0) to suppress occasional huge gaps
        # that destabilise random-init attention.
        self.iat_clip = 5.0

        # One learnable decay parameter per attention head.
        # Initialise so softplus(.) ≈ small positive (gentle decay) at start.
        self.head_log_lambda = nn.Parameter(torch.full((nhead,), -2.0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def _iat_positional_encoding(self, timestamps):
        # timestamps: (B, L) cumulative seconds within window
        B, L = timestamps.shape
        d = self.d_model
        half = d // 2
        device = timestamps.device
        div = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half, 1)
        )
        angle = timestamps.unsqueeze(-1) * div  # (B, L, half)
        pe = torch.zeros(B, L, d, device=device, dtype=torch.float32)
        pe[..., 0:half] = torch.sin(angle)
        pe[..., half:2 * half] = torch.cos(angle)
        return pe

    def forward(self, x, timestamps=None, key_padding_mask=None):
        # x: (B, L, input_size); timestamps: (B, L); key_padding_mask: (B, L) True=pad
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, L, _ = x.shape
        device = x.device

        if timestamps is None:
            # Fallback: treat per-packet IAT field (index 0) as IAT, take cumsum.
            timestamps = torch.cumsum(x[..., 0].float(), dim=1)

        # Clip the raw IAT input feature (index 0) to a sane range.
        x = x.float().clone()
        x[..., 0] = x[..., 0].clamp(min=0.0, max=self.iat_clip)
        # Also clip cumulative timestamps used by PE — they should already be
        # bounded by the dataset's t_max, but defend against pcap clock skew.
        timestamps = timestamps.clamp(min=0.0, max=max(self.iat_clip * 100, 600.0))

        h = self.input_proj(x)
        h = self.input_norm(h)
        h = h + self._iat_positional_encoding(timestamps)

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, L+1, d)

        if key_padding_mask is not None:
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
            kpm = torch.cat([cls_pad, key_padding_mask], dim=1)
        else:
            kpm = None

        # Decay bias — per-head, broadcast over query dimension.
        if kpm is not None:
            ts_for_max = timestamps.masked_fill(key_padding_mask, float('-inf'))
        else:
            ts_for_max = timestamps
        t_max = ts_for_max.max(dim=1, keepdim=True).values  # (B, 1)
        # Guard against batches where all rows are padded (shouldn't happen).
        t_max = torch.where(torch.isfinite(t_max), t_max, torch.zeros_like(t_max))
        delta = (t_max - timestamps).clamp(min=0.0)  # (B, L)
        delta_full = torch.cat([torch.zeros(B, 1, device=device), delta], dim=1)  # (B, L+1)

        lam = nn.functional.softplus(self.head_log_lambda)  # (nhead,)
        # bias[b, h, i, j] = -lam[h] * delta_full[b, j]
        bias = -lam.view(1, self.nhead, 1, 1) * delta_full.view(B, 1, 1, L + 1)
        bias = bias.expand(B, self.nhead, L + 1, L + 1).contiguous()
        # Merge key-padding mask into attention bias (avoids mixed-type warning).
        if kpm is not None:
            pad_bias = torch.zeros_like(kpm, dtype=bias.dtype).masked_fill(kpm, float('-inf'))
            bias = bias + pad_bias.view(B, 1, 1, L + 1)
        attn_mask = bias.view(B * self.nhead, L + 1, L + 1)

        out = self.encoder(h, mask=attn_mask)
        cls_out = self.dropout(out[:, 0, :])
        out = self.head(cls_out)

        if self.activation == 'relu':
            return torch.relu(out)
        elif self.activation == 'softplus':
            return torch.nn.functional.softplus(out)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'exp':
            return torch.exp(out)
        return out


class ModelFactory:
    _registry = {
        'rnn': RNN,
        'lstm': LSTM,
        'bilstm': BiLSTM,
        'gru': GRU,
        'bigru': BiGRU,
        'transformer': TransformerRoFex,
    }

    @staticmethod
    def create_model(model_type, input_size, hidden_size, num_layers, output_size,
                     activation='none', nhead=4, dim_feedforward=256):
        model_type = model_type.lower()
        if model_type not in ModelFactory._registry:
            raise ValueError(f"Unsupported model type: {model_type}. Available: {list(ModelFactory._registry)}")
        if model_type == 'transformer':
            print(f"Creating TRANSFORMER model — input={input_size}, d_model={hidden_size}, "
                  f"layers={num_layers}, nhead={nhead}, ff={dim_feedforward}, output={output_size}")
            return TransformerRoFex(input_size, hidden_size, num_layers, output_size,
                                    activation=activation, nhead=nhead, dim_feedforward=dim_feedforward)
        print(f"Creating {model_type.upper()} model — input={input_size}, hidden={hidden_size}, layers={num_layers}, output={output_size}")
        return ModelFactory._registry[model_type](input_size, hidden_size, num_layers, output_size, activation)


class DataManager:
    @staticmethod
    def load_and_normalize_features(csv_path: str, scaler_save_path: str | None = None):
        """Load AfterImage CSV and normalize per-feature to [-1, 1] via min-max.

        Saves the (min, max) scaler to scaler_save_path if provided so prediction
        outputs can be compared in the same space.  Returns (DataFrame, (feat_min, feat_max)).
        """
        print(f"Loading features from: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        features = pd.read_csv(csv_path, low_memory=False)
        features = features.drop(columns=[c for c in features.columns if str(c).lower() == 'label'], errors='ignore')
        features = features.apply(pd.to_numeric, errors='coerce')
        print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")

        feat_min = features.min()
        feat_max = features.max()
        denom = (feat_max - feat_min).replace(0, 1)
        normalized = 2 * (features - feat_min) / denom - 1  # → [-1, 1]
        normalized = normalized.fillna(0)

        if scaler_save_path is not None:
            import pickle
            with open(scaler_save_path, 'wb') as f:
                pickle.dump({'feat_min': feat_min.values, 'feat_max': feat_max.values}, f)
            print(f"  Scaler saved to {scaler_save_path}")

        print(f"  Normalized range: [{normalized.values.min():.3f}, {normalized.values.max():.3f}]")
        return normalized, (feat_min, feat_max)

    @staticmethod
    def create_dataloader(dataset, batch_size: int) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)


class Trainer:
    def __init__(self, model: nn.Module, config: ModelConfig, artifact_dir: str, model_path: str, lipschitz_lambda: float = 1.0):
        self.model_path = Path(model_path) if model_path else None
        self.device = torch.device(config.device)
        self.epochs = config.epochs
        self.lipschitz_lambda = lipschitz_lambda

        if self.model_path is None:
            self.model = model
            self.config = config
            self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            self.model = None
            self.config = None
            self.load_model()

        self.artifact_dir = Path(artifact_dir)
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.writer = SummaryWriter(log_dir=self.artifact_dir / "tensorboard_logs")

        print(f"Trainer initialized — device={self.device}, lr={config.learning_rate}, wd={config.weight_decay}")

    def load_model(self) -> None:
        print(f"Loading model from: {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'config' in checkpoint:
            self.config = ModelConfig()
            self.config.from_dict(checkpoint['config'])
        else:
            raise ValueError("No configuration found in checkpoint.")
        self.model = ModelFactory.create_model(
            self.config.model_type, self.config.input_size, self.config.hidden_size,
            self.config.num_layers, self.config.output_size, getattr(self.config, 'activation', 'none'),
            nhead=getattr(self.config, 'nhead', 4),
            dim_feedforward=getattr(self.config, 'dim_feedforward', 256),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.to(self.device)
        self.config.epochs = self.epochs
        print(f"  ✓ Model loaded: {self.config.model_type.upper()} on {self.device}")

    def lipschitz_loss(self, p, p_adv, target, outputs, outputs_adv, lamda=1.0):
        ll = torch.mean(((outputs - outputs_adv).norm(p=2)) / ((p - p_adv).norm(p=2) + 1e-8))
        return lamda * ll + self.criterion(outputs, target)

    def train_epoch(self, dataloader: DataLoader, epoch: int, scheduler=None) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=num_batches,
                    desc=f"Epoch {epoch+1:03d}/{self.config.epochs:03d}", leave=False)

        is_transformer = self.config.model_type.lower() == 'transformer'

        for batch_idx, batch in pbar:
            if is_transformer:
                packets, packets_adv, timestamps, key_pad_mask, target = batch
                packets = packets.to(self.device, non_blocking=True)
                packets_adv = packets_adv.to(self.device, non_blocking=True)
                timestamps = timestamps.to(self.device, non_blocking=True)
                key_pad_mask = key_pad_mask.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                outputs = self.model(packets, timestamps=timestamps, key_padding_mask=key_pad_mask)
                outputs_adv = self.model(packets_adv, timestamps=timestamps, key_padding_mask=key_pad_mask)
            else:
                packets, packets_adv, target = batch
                packets = packets.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                packets_adv = packets_adv.to(self.device, non_blocking=True)

                outputs = self.model(packets)
                outputs_adv = self.model(packets_adv)

            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            if len(outputs_adv.shape) == 1:
                outputs_adv = outputs_adv.unsqueeze(0)
            if len(target.shape) == 1:
                target = target.unsqueeze(0)

            loss = self.lipschitz_loss(packets, packets_adv, target, outputs, outputs_adv, lamda=self.lipschitz_lambda)

            if not torch.isfinite(loss):
                if batch_idx < 3:  # only log first few to avoid spam
                    out_finite = torch.isfinite(outputs).all().item()
                    in_finite = torch.isfinite(packets).all().item()
                    p_diff = (packets - packets_adv).norm().item()
                    o_diff = (outputs - outputs_adv).norm().item()
                    print(f"  [skip non-finite loss] batch={batch_idx} "
                          f"in_finite={in_finite} out_finite={out_finite} "
                          f"|p-p_adv|={p_diff:.3e} |o-o_adv|={o_diff:.3e}")
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            pbar.set_postfix({
                'Loss': f'{batch_loss:.6f}',
                'Avg': f'{total_loss/(batch_idx+1):.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            })

            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/Batch', batch_loss, epoch * num_batches + batch_idx)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch * num_batches + batch_idx)

        return total_loss / num_batches

    def train(self, pcap_path: str, csv_path: str) -> Dict[str, Any]:
        print("=" * 50)
        print("STARTING TRAINING")
        print("=" * 50)

        scaler_path = str(self.artifact_dir / "afterimage_scaler.pkl")
        features, _ = DataManager.load_and_normalize_features(csv_path, scaler_save_path=scaler_path)
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
        if PcapDataset is None:
            raise ImportError("PcapDataset not available. Check datasets.py import.")

        num_samples = len(features)
        print(f"Using {num_samples} samples for training")
        if self.config.model_type.lower() == 'transformer':
            if PcapTimeWindowedDataset_lipschitz is None:
                raise ImportError("PcapTimeWindowedDataset_lipschitz not available — check datasets.py.")
            print(f"  Time-windowed dataset: t_max={self.config.t_max}s, n_max={self.config.n_max} packets")
            dataset = PcapTimeWindowedDataset_lipschitz(
                pcap_file=pcap_path, features=features, max_iterations=num_samples,
                t_max=self.config.t_max, n_max=self.config.n_max,
            )
            dataloader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True,
                collate_fn=time_windowed_collate, num_workers=0, pin_memory=True,
            )
        else:
            dataset = PcapDataset(pcap_file=pcap_path, features=features,
                                  max_iterations=num_samples, window_size=self.config.seq_len)
            dataloader = DataManager.create_dataloader(dataset, self.config.batch_size)

        warmup_steps = self.config.warmup_epochs * len(dataloader)
        total_steps = self.config.epochs * len(dataloader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        print(f"Epochs={self.config.epochs}, batch={self.config.batch_size}, "
              f"lr={self.config.learning_rate}, warmup={self.config.warmup_epochs} epochs, "
              f"patience={self.config.patience}")

        logs_dir = self.artifact_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        epoch_logs_path = logs_dir / "epoch_logs"

        start_time = time.time()
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch, scheduler=scheduler)
            self.train_losses.append(avg_loss)
            epoch_time = time.time() - epoch_start

            self.writer.add_scalar('Loss/Epoch', avg_loss, epoch)
            self.writer.add_scalar('Time/Epoch', epoch_time, epoch)

            epoch_log_str = (f"Epoch {epoch+1:03d}/{self.config.epochs:03d} | "
                             f"Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s | Best: {min(self.train_losses):.6f}")
            print(epoch_log_str)
            with open(epoch_logs_path, "a") as log_file:
                log_file.write(epoch_log_str + "\n")

            if avg_loss < best_loss - self.config.min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'config': self.config.to_dict()
                }, self.artifact_dir / "best_model.pt")
                print(f"  ✓ New best model saved (loss: {best_loss:.6f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        total_time = time.time() - start_time
        print("=" * 50)
        print(f"TRAINING COMPLETED — time={total_time:.2f}s, best_loss={best_loss:.6f}")
        print("=" * 50)
        self.writer.close()

        return {'final_loss': self.train_losses[-1], 'best_loss': best_loss,
                'total_time': total_time, 'losses': self.train_losses}

    def save_artifacts(self) -> None:
        print("\nSaving training artifacts...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'losses': self.train_losses
        }, self.artifact_dir / "final_model.pt")
        print(f"  ✓ Final model saved")

        loss_df = pd.DataFrame({'epoch': range(1, len(self.train_losses) + 1), 'loss': self.train_losses})
        loss_df.to_csv(self.artifact_dir / "training_losses.csv", index=False)

        with open(self.artifact_dir / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self.create_training_plots()

    def create_training_plots(self) -> None:
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Results - {self.config.model_type.upper()}', fontsize=16)

        epochs_range = range(1, len(self.train_losses) + 1)
        axes[0, 0].plot(epochs_range, self.train_losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(self.train_losses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Loss Distribution')
        axes[0, 1].set_xlabel('Loss')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].semilogy(epochs_range, self.train_losses, 'r-', linewidth=2)
        axes[1, 0].set_title('Training Loss (Log Scale)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (log)')
        axes[1, 0].grid(True, alpha=0.3)

        stats_text = (
            f"Model: {self.config.model_type.upper()}\n"
            f"Final Loss: {self.train_losses[-1]:.6f}\n"
            f"Best Loss: {min(self.train_losses):.6f}\n"
            f"Total Epochs: {len(self.train_losses)}\n\n"
            f"Input: {self.config.input_size}  Hidden: {self.config.hidden_size}\n"
            f"Layers: {self.config.num_layers}  LR: {self.config.learning_rate}\n"
            f"Weight Decay: {self.config.weight_decay}"
        )
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Training Summary')

        plt.tight_layout()
        plot_path = self.artifact_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Training plots saved: {plot_path}")


class Evaluator:
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.model = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._compiled = False

    def load_model(self) -> None:
        print(f"Loading model from: {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'config' in checkpoint:
            self.config = ModelConfig()
            self.config.from_dict(checkpoint['config'])
        elif self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = ModelConfig()
                self.config.from_dict(json.load(f))
        else:
            raise ValueError("No configuration found. Provide config.json or use a full checkpoint.")
        self.model = ModelFactory.create_model(
            self.config.model_type, self.config.input_size, self.config.hidden_size,
            self.config.num_layers, self.config.output_size, getattr(self.config, 'activation', 'none'),
            nhead=getattr(self.config, 'nhead', 4),
            dim_feedforward=getattr(self.config, 'dim_feedforward', 256),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ Model loaded: {self.config.model_type.upper()} on {self.device}")

    def maybe_compile(self) -> None:
        if self.model is None or self._compiled:
            return
        self.model = torch.compile(self.model)
        self._compiled = True

    def predict_batch(
        self,
        pcap_path: str,
        output_path: str | None = None,
        *,
        batch_size: int = 512,
        seq_len: int | None = None,
        limit_packets: int | None = None,
        amp: bool = False,
        compile_model: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Stream PCAP -> windows -> batched model inference; optionally write CSV incrementally."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()
        if compile_model:
            self.maybe_compile()

        if seq_len is None:
            seq_len = int(getattr(self.config, "seq_len", 1) or 1)
        if seq_len <= 0:
            raise ValueError(f"Invalid seq_len={seq_len}")

        feature_dim = int(getattr(self.config, "input_size", 0) or 0)
        if feature_dim <= 0:
            raise ValueError("Invalid/unknown input_size in config.")

        from scapy.utils import PcapReader
        from preprocessing import FeatureRepresentation

        zero_emb = torch.zeros((feature_dim,), dtype=torch.float32)

        def _fix_dim(v: torch.Tensor) -> torch.Tensor:
            v = v.detach().to(dtype=torch.float32)
            if v.ndim != 1:
                v = v.reshape(-1)
            n = int(v.numel())
            if n == feature_dim:
                return v
            if n > feature_dim:
                return v[:feature_dim]
            out = torch.zeros((feature_dim,), dtype=torch.float32)
            out[:n] = v
            return out

        autocast_enabled = bool(amp and self.device.type == "cuda")
        autocast_dtype = None
        if autocast_enabled:
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        out_path_obj: Path | None = Path(output_path) if output_path else None
        if out_path_obj is not None:
            out_path_obj.parent.mkdir(parents=True, exist_ok=True)

        f_out = None
        writer = None
        header_written = False
        predictions_in_memory: list[np.ndarray] = []

        fr = FeatureRepresentation()
        window: deque[torch.Tensor] = deque(maxlen=seq_len)
        batch_windows: list[torch.Tensor] = []

        def _flush_batch() -> None:
            nonlocal f_out, writer, header_written
            if not batch_windows:
                return
            x = torch.stack(batch_windows, dim=0).to(self.device, non_blocking=True)
            batch_windows.clear()
            with torch.inference_mode():
                if autocast_enabled:
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        y = self.model(x)
                else:
                    y = self.model(x)
            y_np = y.detach().to(dtype=torch.float32, device="cpu").numpy()
            if y_np.ndim == 1:
                y_np = y_np.reshape(-1, 1)

            if out_path_obj is None:
                for row in y_np:
                    predictions_in_memory.append(row.astype(np.float32, copy=False))
                return

            if writer is None:
                import csv
                f_out = out_path_obj.open("w", newline="")
                writer = csv.writer(f_out)

            if not header_written:
                writer.writerow([f"feature_{i+1}" for i in range(y_np.shape[1])])
                header_written = True
            for row in y_np:
                writer.writerow(row.tolist())

        iterator = PcapReader(str(pcap_path))
        try:
            pbar = tqdm(desc="Predicting", unit="pkt") if show_progress else None
            prev_pkt = None
            for idx, pkt in enumerate(iterator):
                if limit_packets is not None and idx >= limit_packets:
                    break
                if prev_pkt is None:
                    prev_pkt = pkt
                emb = fr.get_int_embedded_representation(pkt, prev_pkt, adv=False)
                prev_pkt = pkt
                emb_fixed = _fix_dim(emb) if emb is not None else zero_emb

                window.append(emb_fixed)
                if len(window) < seq_len:
                    pad = torch.zeros((seq_len - len(window), feature_dim), dtype=torch.float32)
                    x_win = torch.cat([pad, torch.stack(list(window), dim=0)], dim=0)
                else:
                    x_win = torch.stack(list(window), dim=0)
                batch_windows.append(x_win)
                if len(batch_windows) >= batch_size:
                    _flush_batch()
                if pbar is not None:
                    pbar.update(1)
            _flush_batch()
            if pbar is not None:
                pbar.close()
        finally:
            try:
                iterator.close()
            except Exception:
                pass
            if f_out is not None:
                f_out.close()

        if out_path_obj is not None:
            print(f"✓ Predictions saved to: {out_path_obj}")
            return pd.DataFrame()

        predictions_array = np.vstack(predictions_in_memory) if predictions_in_memory else np.empty((0, 0), dtype=np.float32)
        return pd.DataFrame(predictions_array, columns=[f"feature_{i+1}" for i in range(predictions_array.shape[1])])


_SRC_DIR = Path(__file__).parent
_REPO_ROOT = _SRC_DIR.parent
_DEFAULT_CHECKPOINTS_ROOT = _REPO_ROOT / "checkpoints"
_DEFAULT_DATA_ROOT = _REPO_ROOT / "data"


def infer_output_from_pcap(pcap_path: Path, data_root: Path, features_subdir: str = 'rofex') -> Path:
    """Infer the feature output CSV path from a pcap path under data_root.

    data/<dataset>/pcaps/<split>/[<variant>/]<attack>.pcap
      ->  data/<dataset>/features/<features_subdir>/<split>/[<variant>/]<attack>.csv
    """
    rel = pcap_path.resolve().relative_to(data_root.resolve())
    # rel.parts: (dataset, 'pcaps', split, [variant,] attack.pcap)
    parts = list(rel.parts)
    try:
        pcaps_idx = parts.index('pcaps')
    except ValueError:
        raise ValueError(
            f"Cannot infer output path: 'pcaps' not found in {pcap_path} relative to data_root {data_root}. "
            "Use --output-csv to specify explicitly."
        )
    new_parts = parts[:pcaps_idx] + ['features', features_subdir] + parts[pcaps_idx + 1:]
    return (data_root / Path(*new_parts)).with_suffix('.csv')


def setup_logging(artifact_dir: Path) -> logging.Logger:
    log_dir = artifact_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def create_checkpoint_dir(checkpoints_root: Path, dataset: str, model_type: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = checkpoints_root / dataset / f"{model_type}_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("plots", "logs", "tensorboard_logs"):
        (artifact_dir / subdir).mkdir(exist_ok=True)
    return artifact_dir


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoFex: Lipschitz-constrained feature extractor training/prediction")

    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train')
    parser.add_argument('--dataset', type=str, choices=['kitsune', 'x-iot', 'kitsune-raw'], required=True,
                        help='Dataset name — determines checkpoint and data subdirectory.')

    # Data paths
    parser.add_argument('--pcap-path', type=str, default=None,
                        help='Predict (single): path to a single .pcap file under data-root.')
    parser.add_argument('--pcap-folder', type=str, default=None,
                        help='Predict (batch): path to a folder; all .pcap files inside are processed.')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='AfterImage CSV (required for train mode).')

    # Model architecture
    parser.add_argument('--model-type', type=str,
                        choices=['rnn', 'lstm', 'bilstm', 'gru', 'bigru', 'transformer'], default='lstm')
    parser.add_argument('--input-size', type=int, default=144)
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='RNN hidden size, or Transformer d_model (must be divisible by --nhead).')
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--output-size', type=int, default=100)
    parser.add_argument('--seq-len', type=int, default=5,
                        help='RNN-only: fixed sliding window size. Ignored for transformer.')

    # Transformer-specific
    parser.add_argument('--nhead', type=int, default=4,
                        help='Transformer-only: number of attention heads.')
    parser.add_argument('--dim-feedforward', type=int, default=256,
                        help='Transformer-only: feedforward dimension inside encoder layers.')
    parser.add_argument('--t-max', type=float, default=10.0,
                        help='Transformer-only: max time window in seconds for variable-length context.')
    parser.add_argument('--n-max', type=int, default=512,
                        help='Transformer-only: max packets per window (cap for high-rate flooding cases).')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Linear LR warmup duration in epochs, then cosine decay.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience in epochs.')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Early stopping min loss improvement threshold.')
    parser.add_argument('--lipschitz-lambda', type=float, default=1.0,
                        help='Weight on the Lipschitz penalty term (default 1.0; lower = less constraint).')
    parser.add_argument('--activation', type=str, choices=['none', 'relu', 'softplus', 'sigmoid', 'exp'], default='none')
    parser.add_argument('--device', type=str, default='auto')

    # Storage roots (resolved via repo symlinks by default)
    parser.add_argument('--checkpoints-root', type=str, default=str(_DEFAULT_CHECKPOINTS_ROOT),
                        help='Root for saving checkpoints → <root>/<dataset>/<model>_<timestamp>/')
    parser.add_argument('--data-root', type=str, default=str(_DEFAULT_DATA_ROOT),
                        help='Root of the organised data folder. Output features written to '
                             '<root>/<dataset>/features/rofex/... mirroring pcaps layout.')

    # Predict-specific
    parser.add_argument('--model-path', type=str, default=None, help='Path to best_model.pt')
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Predict (single, override): explicit output CSV path, bypasses inferred layout.')
    parser.add_argument('--predict-batch-size', type=int, default=512)
    parser.add_argument('--predict-seq-len', type=int, default=None)
    parser.add_argument('--predict-limit-packets', type=int, default=None)
    parser.add_argument('--predict-amp', action='store_true')
    parser.add_argument('--predict-compile', action='store_true')
    parser.add_argument('--features-subdir', type=str, default='rofex',
                        help='Subdirectory under features/ for predict outputs '
                             '(e.g. "rofex", "rofex-transformer"). Mirrors pcaps layout.')

    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (args.device if args.device != 'auto' else 'cpu')

    print(f"Mode: {args.mode.upper()} | Dataset: {args.dataset} | Device: {device}")

    if args.mode == 'train':
        if not args.pcap_path or not args.csv_path:
            print("Error: --pcap-path and --csv-path are required for train mode")
            sys.exit(1)

        config = ModelConfig()
        config.model_type = args.model_type
        config.input_size = args.input_size
        config.hidden_size = args.hidden_size
        config.num_layers = args.num_layers
        config.output_size = args.output_size
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.weight_decay = args.weight_decay
        config.activation = args.activation
        config.device = device
        config.seq_len = args.seq_len
        config.nhead = args.nhead
        config.dim_feedforward = args.dim_feedforward
        config.t_max = args.t_max
        config.n_max = args.n_max
        config.patience = args.patience
        config.min_delta = args.min_delta
        config.warmup_epochs = args.warmup_epochs

        artifact_dir = create_checkpoint_dir(Path(args.checkpoints_root), args.dataset, args.model_type)
        print(f"Checkpoint dir: {artifact_dir}")

        logger = setup_logging(artifact_dir)
        logger.info(f"Starting training: {config.to_dict()}")

        model = ModelFactory.create_model(
            config.model_type, config.input_size, config.hidden_size,
            config.num_layers, config.output_size, config.activation,
            nhead=config.nhead, dim_feedforward=config.dim_feedforward,
        )
        trainer = Trainer(model, config, artifact_dir, args.model_path, lipschitz_lambda=args.lipschitz_lambda)
        training_results = trainer.train(args.pcap_path, args.csv_path)
        trainer.save_artifacts()
        logger.info(f"Training completed: {training_results}")
        print(f"\n✓ Checkpoint saved to: {artifact_dir}")

    elif args.mode == 'predict':
        if not args.model_path:
            print("Error: --model-path is required for predict mode")
            sys.exit(1)
        if not args.pcap_path and not args.pcap_folder:
            print("Error: provide --pcap-path (single) or --pcap-folder (batch)")
            sys.exit(1)

        evaluator = Evaluator(args.model_path, args.config_path)
        evaluator.load_model()

        predict_kwargs = dict(
            batch_size=args.predict_batch_size,
            seq_len=args.predict_seq_len,
            limit_packets=args.predict_limit_packets,
            amp=args.predict_amp,
            compile_model=args.predict_compile,
        )

        data_root = Path(args.data_root)

        if args.pcap_path:
            # Single mode
            if args.output_csv:
                output_path = Path(args.output_csv)
            else:
                output_path = infer_output_from_pcap(Path(args.pcap_path), data_root, args.features_subdir)
            print(f"  {Path(args.pcap_path).name} → {output_path}")
            evaluator.predict_batch(args.pcap_path, str(output_path), **predict_kwargs)

        else:
            # Batch mode
            pcap_folder = Path(args.pcap_folder)
            pcap_files = sorted(pcap_folder.glob("*.pcap"))
            if not pcap_files:
                print(f"No .pcap files found in {pcap_folder}")
                sys.exit(1)
            print(f"Batch predict: {len(pcap_files)} files in {pcap_folder}")
            for pcap_file in pcap_files:
                output_path = infer_output_from_pcap(pcap_file, data_root, args.features_subdir)
                print(f"  {pcap_file.name} → {output_path}")
                evaluator.predict_batch(str(pcap_file), str(output_path), **predict_kwargs)


if __name__ == "__main__":
    main()
