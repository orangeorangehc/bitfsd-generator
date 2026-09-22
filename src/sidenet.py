"""
Lightweight Transformer SideNet for Left/Right cone classification.

Takes N cone coordinates and predicts Left/Right per cone.
Supports configurable input: (x, y) only, (x, y, z), or (x, y, score).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sin/cos positional encoding for x, y coordinates."""

    def __init__(self, d_model=32, max_freq=8):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

    def forward(self, x, y):
        """
        Args:
            x: (*) x-coordinates in meters
            y: (*) y-coordinates in meters
        Returns:
            (*, d_model)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, 1)
        y_flat = y.reshape(-1, 1)

        encodings = []
        for i in range(self.max_freq):
            freq = math.pi * (2 ** i)
            encodings.append(torch.sin(freq * x_flat))
            encodings.append(torch.cos(freq * x_flat))
            encodings.append(torch.sin(freq * y_flat))
            encodings.append(torch.cos(freq * y_flat))

        enc = torch.cat(encodings, dim=-1)
        if enc.shape[-1] > self.d_model:
            enc = enc[:, :self.d_model]
        elif enc.shape[-1] < self.d_model:
            enc = F.pad(enc, (0, self.d_model - enc.shape[-1]))

        if len(orig_shape) > 1:
            return enc.reshape(*orig_shape, self.d_model)
        return enc.reshape(-1, self.d_model)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, key_padding_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SideNet(nn.Module):
    """
    Transformer-based Left/Right classifier for detected cones.

    Args:
        input_mode: 'xy' (2 dims), 'xyz' (3 dims), 'xyzs' (4 dims with score)
        d_model: token dimension (default 64)
        nhead: attention heads (default 4)
        num_layers: encoder blocks (default 3)
        dim_feedforward: FFN hidden dim (default 128)
        dropout: attention dropout (default 0.1)
    """

    INPUT_DIMS = {"xy": 2, "xyz": 3, "xyzs": 4}

    def __init__(self, input_mode="xyz", d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_mode = input_mode
        self.d_model = d_model
        input_dim = self.INPUT_DIMS.get(input_mode, 4)

        # Position encoding on x, y
        self.pos_enc = PositionalEncoding(d_model=32)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
        )
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output head
        self.head = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, points, return_logits=True):
        """
        Args:
            points: (N, D) or (B, N, D) tensor where D depends on input_mode
            return_logits: if True return logits, else probabilities
        Returns:
            (N, 2) or (B, N, 2)
        """
        batched = points.dim() == 3
        if not batched:
            points = points.unsqueeze(0)
        B, N, D = points.shape

        # Extract x, y (always first two)
        x, y = points[..., 0], points[..., 1]

        # Position encoding
        pos = self.pos_enc(x, y)  # (B, N, 32)

        # Input features
        feat = self.input_proj(points)  # (B, N, 32)
        tokens = torch.cat([feat, pos], dim=-1)  # (B, N, 64)
        tokens = self.input_norm(tokens)

        # Key padding mask for zero-padded cones
        key_padding_mask = None
        if N > 0:
            is_pad = (points.abs().sum(dim=-1) == 0)
            if is_pad.any():
                key_padding_mask = is_pad

        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask)

        logits = self.head(tokens)  # (B, N, 2)

        if not batched:
            logits = logits.squeeze(0)

        if not return_logits:
            return F.softmax(logits, dim=-1)
        return logits

    def predict(self, points):
        """Convenience: return class labels (0=Left, 1=Right)."""
        logits = self.forward(points)
        return logits.argmax(dim=-1)

    def predict_probs(self, points):
        """Return softmax probabilities (N, 2)."""
        return self.forward(points, return_logits=False)
