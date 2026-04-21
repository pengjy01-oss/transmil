"""Aggregator modules for MIL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransMILAggregator(nn.Module):
    """TransMIL-style aggregator: CLS token + Transformer self-attention."""

    def __init__(self, in_dim, num_heads=8, num_layers=2, dropout=0.1):
        super(TransMILAggregator, self).__init__()
        if in_dim % num_heads != 0:
            raise ValueError(
                'TransMILAggregator: in_dim ({}) must be divisible by num_heads ({})'.format(in_dim, num_heads))
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(in_dim)
        self._last_attn_weights = None  # [K] CLS->instance attention
        self._need_attn = False

    def forward(self, x):
        x = x.unsqueeze(0)  # [1, K, in_dim]
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [1, 1, in_dim]
        x = torch.cat([cls_token, x], dim=1)  # [1, K+1, in_dim]
        self._last_attn_weights = None
        x = self.encoder(x)  # [1, K+1, in_dim]
        z = self.norm(x[:, 0, :])  # [1, in_dim]
        return z

    @torch.no_grad()
    def compute_attention_weights(self, x):
        """Separate pass to compute CLS->instance attention. Call after forward()."""
        x = x.unsqueeze(0)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        # Run through all layers except last
        for layer in self.encoder.layers[:-1]:
            x = layer(x)
        # For last layer, manually call self_attn with need_weights=True
        last_layer = self.encoder.layers[-1]
        # Pre-norm (if norm_first)
        if last_layer.norm_first:
            x_normed = last_layer.norm1(x)
        else:
            x_normed = x
        _, attn_w = last_layer.self_attn(
            x_normed, x_normed, x_normed,
            need_weights=True, average_attn_weights=True
        )
        # attn_w: [1, K+1, K+1]. CLS row=0, skip CLS col=0
        self._last_attn_weights = attn_w[0, 0, 1:]  # [K]
        return self._last_attn_weights

    def get_last_attention(self):
        """Return CLS->instance attention weights [1, K] or None."""
        if self._last_attn_weights is not None:
            return self._last_attn_weights.unsqueeze(0).detach()
        return None


# ---------------------------------------------------------------------------
# Nystrom-based efficient attention (O(N*m) instead of O(N²))
# ---------------------------------------------------------------------------

class NystromAttention(nn.Module):
    """Nystrom approximation of multi-head self-attention."""

    def __init__(self, dim, num_heads=8, num_landmarks=64, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def _to_heads(self, x):
        B, N, _ = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        B, N, C = x.shape
        q = self._to_heads(self.q_proj(x)) * self.scale
        k = self._to_heads(self.k_proj(x))
        v = self._to_heads(self.v_proj(x))

        m = min(self.num_landmarks, N)

        if N <= m * 2:
            # Standard attention for small sequences
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            out = self.attn_drop(attn) @ v
        else:
            # Nystrom approximation via segment-mean landmarks
            seg = (N + m - 1) // m
            pad = seg * m - N
            if pad > 0:
                q_p = F.pad(q, (0, 0, 0, pad))
                k_p = F.pad(k, (0, 0, 0, pad))
            else:
                q_p, k_p = q, k
            q_land = q_p.reshape(B, self.num_heads, m, seg, self.head_dim).mean(3)
            k_land = k_p.reshape(B, self.num_heads, m, seg, self.head_dim).mean(3)

            ker1 = (q @ k_land.transpose(-2, -1)).softmax(-1)        # [B,H,N,m]
            ker2 = (q_land @ k_land.transpose(-2, -1)).softmax(-1)  # [B,H,m,m]
            ker3 = (q_land @ k.transpose(-2, -1)).softmax(-1)       # [B,H,m,N]

            ker2_inv = torch.linalg.pinv(ker2.float()).to(ker2.dtype)
            out = self.attn_drop(ker1) @ (ker2_inv @ (ker3 @ v))

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

    def compute_full_attention(self, x):
        """Full O(N^2) attention for interpretability."""
        B, N, C = x.shape
        q = self._to_heads(self.q_proj(x)) * self.scale
        k = self._to_heads(self.k_proj(x))
        attn = (q @ k.transpose(-2, -1)).softmax(-1)
        return attn.mean(dim=1)  # [B, N, N]


class _NystromEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with Nystrom attention."""

    def __init__(self, dim, num_heads=8, num_landmarks=64, dropout=0.1):
        super().__init__()
        self.self_attn = NystromAttention(dim, num_heads, num_landmarks, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


class NystromTransMILAggregator(nn.Module):
    """TransMIL aggregator with Nystrom efficient attention."""

    def __init__(self, in_dim, num_heads=8, num_layers=2, num_landmarks=64, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        self.layers = nn.ModuleList([
            _NystromEncoderLayer(in_dim, num_heads, num_landmarks, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(in_dim)
        self._last_attn_weights = None

    def forward(self, x):
        x = x.unsqueeze(0)  # [1, K, in_dim]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # [1, K+1, in_dim]
        for layer in self.layers:
            x = layer(x)
        z = self.norm(x[:, 0, :])  # [1, in_dim]
        return z

    @torch.no_grad()
    def compute_attention_weights(self, x):
        """Compute CLS->instance attention using full attention on last layer."""
        x = x.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for layer in self.layers[:-1]:
            x = layer(x)
        last = self.layers[-1]
        x_normed = last.norm1(x)
        attn = last.self_attn.compute_full_attention(x_normed)  # [1, K+1, K+1]
        self._last_attn_weights = attn[0, 0, 1:]  # [K]
        return self._last_attn_weights

    def get_last_attention(self):
        """Return CLS->instance attention weights [1, K] or None."""
        if self._last_attn_weights is not None:
            return self._last_attn_weights.unsqueeze(0).detach()
        return None
