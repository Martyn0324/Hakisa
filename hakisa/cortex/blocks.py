"""Building blocks of Hakisa's visual cortex (all trained from scratch).

* ``FusedMBConv`` (EfficientNetV2): a plain 3x3 conv expansion — fastest choice at high resolution.
* ``UIB`` (MobileNetV4's Universal Inverted Bottleneck, ExtraDW variant): depthwise conv before and
  after the 1x1 expansion -> large receptive field for very few FLOPs.
* ``ECA`` (ECA-Net): channel attention with a 1-D conv, ~a dozen parameters per layer.
* ``TransformerBlock``: global self-attention on the 1/16 grid (where are the gaps between bullets?).

Hakisa 1.0 used 200x200 kernels "to simulate attention in image regions"; here attention is real and
the convolutions are small and depthwise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1, act: bool = True):
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if act:
            layers.append(nn.Hardswish())
        super().__init__(*layers)


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020)."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3), keepdim=False).unsqueeze(1)  # (B, 1, C)
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class FusedMBConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, expand: int = 2):
        super().__init__()
        mid = cin * expand
        self.expand = ConvBNAct(cin, mid, 3, stride)
        self.project = ConvBNAct(mid, cout, 1, act=False)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.expand(x))
        return x + y if self.residual else y


class UIB(nn.Module):
    """Universal Inverted Bottleneck (ExtraDW): dw(k_start) -> 1x1 expand -> dw(k_mid, stride) -> ECA -> 1x1."""

    def __init__(self, cin: int, cout: int, stride: int = 1, expand: int = 3, k_start: int = 3, k_mid: int = 5, eca: bool = True):
        super().__init__()
        mid = cin * expand
        self.start_dw = ConvBNAct(cin, cin, k_start, 1, groups=cin, act=False) if k_start else nn.Identity()
        self.expand = ConvBNAct(cin, mid, 1)
        self.mid_dw = ConvBNAct(mid, mid, k_mid, stride, groups=mid)
        self.eca = ECA(mid) if eca else nn.Identity()
        self.project = ConvBNAct(mid, cout, 1, act=False)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.eca(self.mid_dw(self.expand(self.start_dw(x)))))
        return x + y if self.residual else y


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, c))
        return x + self.mlp(self.norm2(x))


class AttentionPool(nn.Module):
    """Learned-query attention pooling (one query) -> global vector; better than flatten->Linear."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(tokens.shape[0], -1, -1)
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(out[:, 0] + tokens.mean(dim=1))


def coord_channels(b: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """CoordConv (Liu et al., 2018): two channels holding x and y in [-1, 1]."""
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([xs, ys], dim=1)


def mlp(din: int, hidden: int, dout: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = din
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    mods.append(nn.Linear(d, dout))
    return nn.Sequential(*mods)
