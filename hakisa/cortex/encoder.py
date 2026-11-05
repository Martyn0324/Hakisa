"""Visual encoder: frames (uint8) -> tokens / feature maps / global vector.

Stages (resolution relative to the input)::

    stem   3x3 conv s2              1/2
    stage1 Fused-MBConv s2          1/4   -> skip (fine spatial detail: bullets, hitbox)
    stage2 UIB s2 (+ UIB)           1/8   -> skip
    stage3 UIB s2 (+ UIB...)        1/16
    attn   Transformer blocks on the 1/16 grid with learned 2-D position embeddings
    pool   learned-query attention pooling -> global vector

Analogy (loose!): stem/stage1 ~ V1 edge/blob detectors, stage2/3 ~ V2/V4 pattern detectors, the
stacked frames give the motion signal that area MT/V5 extracts, and the attention stage integrates
the whole visual field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hakisa.cortex.blocks import AttentionPool, ConvBNAct, FusedMBConv, TransformerBlock, UIB, coord_channels

PRESETS: dict[str, dict[str, int]] = {
    "nano": dict(stem=16, c1=24, c2=40, c3=64, n2=1, n3=2, attn=1, heads=2, expand=3),
    "small": dict(stem=24, c1=32, c2=64, c3=96, n2=2, n3=2, attn=2, heads=4, expand=3),
    "base": dict(stem=32, c1=48, c2=96, c3=160, n2=2, n3=3, attn=2, heads=4, expand=4),
}


@dataclass
class EncoderOutput:
    tokens: torch.Tensor  # (B, N, C) after attention
    fmap: torch.Tensor  # (B, C, h, w) tokens as a map
    skips: list[torch.Tensor]  # [(B, c1, H/4, W/4), (B, c2, H/8, W/8)]
    pooled: torch.Tensor  # (B, C)


def _down(n: int, times: int) -> int:
    for _ in range(times):
        n = (n + 1) // 2
    return n


class VisualEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_hw: tuple[int, int],
        preset: str = "small",
        coord: bool = True,
        film_dim: int = 0,
    ):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r} (choose from {list(PRESETS)})")
        p = PRESETS[preset]
        self.preset = preset
        self.coord = coord
        self.input_hw = input_hw
        cin = in_channels + (2 if coord else 0)
        self.stem = ConvBNAct(cin, p["stem"], 3, 2)
        self.stage1 = FusedMBConv(p["stem"], p["c1"], stride=2, expand=2)
        self.stage2 = nn.Sequential(
            UIB(p["c1"], p["c2"], stride=2, expand=p["expand"]),
            *[UIB(p["c2"], p["c2"], expand=p["expand"]) for _ in range(p["n2"] - 1)],
        )
        self.stage3 = nn.Sequential(
            UIB(p["c2"], p["c3"], stride=2, expand=p["expand"]),
            *[UIB(p["c3"], p["c3"], expand=p["expand"]) for _ in range(p["n3"] - 1)],
        )
        self.grid_hw = (_down(input_hw[0], 4), _down(input_hw[1], 4))
        n_tokens = self.grid_hw[0] * self.grid_hw[1]
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, p["c3"]))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.film = nn.Linear(film_dim, 2 * p["c3"]) if film_dim > 0 else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.blocks = nn.ModuleList([TransformerBlock(p["c3"], p["heads"]) for _ in range(p["attn"])])
        self.norm = nn.LayerNorm(p["c3"])
        self.pool = AttentionPool(p["c3"], p["heads"])
        self.out_dim = p["c3"]
        self.skip_dims = (p["c1"], p["c2"])

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        return (x - 0.5) / 0.25

    def forward(self, x: torch.Tensor, film: torch.Tensor | None = None) -> EncoderOutput:
        x = self.normalize(x)
        if self.coord:
            x = torch.cat([x, coord_channels(x.shape[0], x.shape[2], x.shape[3], x.device, x.dtype)], dim=1)
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        b, c, h, w = s3.shape
        tokens = s3.flatten(2).transpose(1, 2)
        if tokens.shape[1] == self.pos.shape[1]:
            tokens = tokens + self.pos
        else:  # different input size than at construction: interpolate the position grid
            pos = self.pos.transpose(1, 2).reshape(1, c, *self.grid_hw)
            pos = nn.functional.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
            tokens = tokens + pos.flatten(2).transpose(1, 2)
        if self.film is not None and film is not None:
            gamma, beta = self.film(film).unsqueeze(1).chunk(2, dim=-1)
            tokens = tokens * (1 + gamma) + beta
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        fmap = tokens.transpose(1, 2).reshape(b, c, h, w)
        return EncoderOutput(tokens=tokens, fmap=fmap, skips=[s1, s2], pooled=self.pool(tokens))
