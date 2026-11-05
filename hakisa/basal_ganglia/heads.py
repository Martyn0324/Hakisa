"""Action selection ("basal ganglia") and mouse targeting ("superior colliculus").

The basal ganglia are classically described as an action-selection circuit: candidate motor programs
compete and the winner is released from tonic inhibition; dopamine from the substantia nigra / VTA
signals reward-prediction errors that tune that selection (Schultz, Dayan & Montague, Science 1997) —
the biological cousin of the TD error used by the critic during consolidation.

The superior colliculus holds a *topographic motor map*: activity at a location on the map encodes
the vector of the next saccade (Gandhi & Katnani, Annu. Rev. Neurosci. 2011).  Hakisa's pointer head
is the same idea for the mouse: a score map over the screen.

Pointer = Hakisa 1.0's X/Y decomposition, made exact
------------------------------------------------------
The decoder outputs a score s(x, y) per coarse cell plus sub-cell offset logits.  Sampling draws
``x ~ P(x) = sum_y P(x, y)`` (the "X layer") and then ``y ~ P(y | x)`` (the "Y layer").  Unlike two
independent heads, this cannot produce mixed coordinates of two different targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from hakisa.actions import ActionSpace, HeadSpec
from hakisa.cortex.blocks import ConvBNAct


@dataclass
class PointerLogits:
    cell: torch.Tensor  # (B, gh*gw)
    offset: torch.Tensor  # (B, sh*sw, gh*gw)
    grid: tuple[int, int]
    sub: tuple[int, int]

    # -- helpers ---------------------------------------------------------------------------------
    def split(self, fx: torch.Tensor, fy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sh, sw = self.sub
        gw = self.grid[1]
        cell = (fy // sh) * gw + (fx // sw)
        off = (fy % sh) * sw + (fx % sw)
        return cell, off

    def log_prob(self, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        fx = fx.clamp_min(0)
        fy = fy.clamp_min(0)
        cell, off = self.split(fx, fy)
        lp_cell = F.log_softmax(self.cell, dim=-1).gather(1, cell[:, None])[:, 0]
        off_logits = self.offset.gather(2, cell[:, None, None].expand(-1, self.offset.shape[1], 1))[:, :, 0]
        lp_off = F.log_softmax(off_logits, dim=-1).gather(1, off[:, None])[:, 0]
        return lp_cell + lp_off

    def entropy(self) -> torch.Tensor:
        lp_cell = F.log_softmax(self.cell, dim=-1)
        p_cell = lp_cell.exp()
        h_cell = -(p_cell * lp_cell).sum(-1)
        lp_off = F.log_softmax(self.offset, dim=1)
        h_off = -(lp_off.exp() * lp_off).sum(1)  # (B, cells)
        return h_cell + (p_cell * h_off).sum(-1)

    def kl(self, other: "PointerLogits") -> torch.Tensor:
        lp = F.log_softmax(self.cell, dim=-1)
        lq = F.log_softmax(other.cell, dim=-1)
        p = lp.exp()
        kl_cell = (p * (lp - lq)).sum(-1)
        lpo = F.log_softmax(self.offset, dim=1)
        lqo = F.log_softmax(other.offset, dim=1)
        kl_off = (lpo.exp() * (lpo - lqo)).sum(1)
        return kl_cell + (p * kl_off).sum(-1)

    def sample(self, deterministic: bool = False, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """X then Y|X, then the sub-cell offset. Returns fine-grid (fx, fy)."""
        gh, gw = self.grid
        sh, sw = self.sub
        b = self.cell.shape[0]
        joint = self.cell.view(b, gh, gw)
        if deterministic:
            cell = self.cell.argmax(-1)
            cy, cx = cell // gw, cell % gw
        else:
            logp_x = torch.logsumexp(joint, dim=1)  # marginal over rows -> P(x)
            cx = torch.multinomial(F.softmax(logp_x, -1), 1, generator=generator)[:, 0]
            col = joint.gather(2, cx.view(b, 1, 1).expand(b, gh, 1))[:, :, 0]  # scores of column x
            cy = torch.multinomial(F.softmax(col, -1), 1, generator=generator)[:, 0]
            cell = cy * gw + cx
        off_logits = self.offset.gather(2, cell[:, None, None].expand(-1, self.offset.shape[1], 1))[:, :, 0]
        if deterministic:
            off = off_logits.argmax(-1)
        else:
            off = torch.multinomial(F.softmax(off_logits, -1), 1, generator=generator)[:, 0]
        fx = cx * sw + off % sw
        fy = cy * sh + off // sw
        return fx, fy

    def tempered(self, temperature: float) -> "PointerLogits":
        if temperature == 1.0:
            return self
        return PointerLogits(self.cell / temperature, self.offset / temperature, self.grid, self.sub)

    def expected_xy(self) -> torch.Tensor:
        """Expected cell centre in normalised [0,1] coordinates (B, 2) — handy for visualisation."""
        gh, gw = self.grid
        p = F.softmax(self.cell, -1).view(-1, gh, gw)
        ys = (torch.arange(gh, device=p.device, dtype=p.dtype) + 0.5) / gh
        xs = (torch.arange(gw, device=p.device, dtype=p.dtype) + 0.5) / gw
        return torch.stack([(p.sum(1) * xs).sum(-1), (p.sum(2) * ys).sum(-1)], dim=-1)


@dataclass
class PolicyOutput:
    logits: dict[str, torch.Tensor] = field(default_factory=dict)
    pointer: dict[str, PointerLogits] = field(default_factory=dict)
    value: torch.Tensor | None = None
    danger: torch.Tensor | None = None  # logit of "life lost within the horizon"
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


def _cat_entropy(logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits, -1)
    return -(lp.exp() * lp).sum(-1)


def _cat_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(p_logits, -1)
    lq = F.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


class ActionDistribution:
    """Joint distribution over all heads of an :class:`ActionSpace` (product of factors)."""

    def __init__(self, space: ActionSpace, out: PolicyOutput):
        self.space = space
        self.out = out

    # -- activity of pointer heads ---------------------------------------------------------------
    def pointer_active(self, name: str, actions: dict[str, torch.Tensor] | None) -> torch.Tensor:
        users = self.space.pointer_users.get(name) or []
        ref = self.out.pointer[name].cell
        if not users:
            return torch.ones(ref.shape[0], device=ref.device, dtype=ref.dtype)
        if actions is None:  # expected activity under the policy
            prob = torch.zeros(ref.shape[0], device=ref.device, dtype=ref.dtype)
            for head, idx in users:
                prob = prob + F.softmax(self.out.logits[head], -1)[:, idx].sum(-1)
            return prob
        mask = torch.zeros(ref.shape[0], device=ref.device, dtype=torch.bool)
        for head, idx in users:
            a = actions[head]
            for i in idx:
                mask |= a == i
        return mask.to(ref.dtype)

    # -- likelihoods ------------------------------------------------------------------------------
    def log_prob(self, actions: dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for h in self.space.categorical_heads:
            lp = F.log_softmax(self.out.logits[h.name], -1).gather(1, actions[h.name].clamp_min(0)[:, None])[:, 0]
            total = lp if total is None else total + lp
        for p in self.space.pointer_heads:
            lp = self.out.pointer[p.name].log_prob(actions[f"{p.name}_x"], actions[f"{p.name}_y"])
            lp = lp * self.pointer_active(p.name, actions)
            total = lp if total is None else total + lp
        assert total is not None
        return total

    def entropy(self, actions: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        total = None
        for h in self.space.categorical_heads:
            e = _cat_entropy(self.out.logits[h.name])
            total = e if total is None else total + e
        for p in self.space.pointer_heads:
            e = self.out.pointer[p.name].entropy() * self.pointer_active(p.name, actions)
            total = e if total is None else total + e
        assert total is not None
        return total

    def kl(self, other: "ActionDistribution", actions: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Exact KL(self || other), summed over heads (pointer terms weighted by activity)."""
        total = None
        for h in self.space.categorical_heads:
            k = _cat_kl(self.out.logits[h.name], other.out.logits[h.name])
            total = k if total is None else total + k
        for p in self.space.pointer_heads:
            k = self.out.pointer[p.name].kl(other.out.pointer[p.name]) * self.pointer_active(p.name, actions)
            total = k if total is None else total + k
        assert total is not None
        return total

    # -- sampling ---------------------------------------------------------------------------------
    def sample_categorical(self, deterministic: bool = False, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        """``temperature`` < 1 sharpens play-time sampling (RL always samples at 1)."""
        out = {}
        for h in self.space.categorical_heads:
            logits = self.out.logits[h.name]
            if deterministic:
                out[h.name] = logits.argmax(-1)
            else:
                out[h.name] = torch.multinomial(F.softmax(logits / temperature, -1), 1)[:, 0]
        return out

    def sample_pointers(self, deterministic: bool = False, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        out = {}
        for p in self.space.pointer_heads:
            fx, fy = self.out.pointer[p.name].tempered(temperature).sample(deterministic)
            out[f"{p.name}_x"], out[f"{p.name}_y"] = fx, fy
        return out

    def sample(self, deterministic: bool = False, temperature: float = 1.0) -> dict[str, torch.Tensor]:
        out = self.sample_categorical(deterministic, temperature)
        out.update(self.sample_pointers(deterministic, temperature))
        return out

    # -- supervised losses ------------------------------------------------------------------------
    def bc_losses(
        self,
        labels: dict[str, torch.Tensor],
        class_weights: dict[str, torch.Tensor] | None = None,
        label_smoothing: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for h in self.space.categorical_heads:
            w = class_weights.get(h.name) if class_weights else None
            losses[h.name] = F.cross_entropy(
                self.out.logits[h.name],
                labels[h.name],
                weight=w.to(self.out.logits[h.name].device) if w is not None else None,
                ignore_index=-1,
                # never smooth rare edge-triggered commands: a smoothed bomb head learns a ~eps/2 bomb
                # probability *every tick*, i.e. a bomb every couple of seconds at 20 Hz
                label_smoothing=label_smoothing if h.mode == "hold" else 0.0,
            )
        for p in self.space.pointer_heads:
            fx, fy = labels[f"{p.name}_x"], labels[f"{p.name}_y"]
            valid = (fx >= 0) & (fy >= 0)
            if valid.any():
                lp = self.out.pointer[p.name].log_prob(fx, fy)
                losses[p.name] = -(lp * valid).sum() / valid.sum()
            else:
                losses[p.name] = self.out.pointer[p.name].cell.sum() * 0.0
        return losses


# ---------------------------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------------------------


class PointerDecoder(nn.Module):
    """Superior colliculus: U-Net-lite from the 1/16 map back to the 1/4 grid -> cell scores + offsets."""

    def __init__(self, c_top: int, c_skip2: int, c_skip1: int, head: HeadSpec, width: int = 48, cond_dim: int = 0):
        super().__init__()
        self.grid = head.grid
        self.sub = head.sub
        self.lat_top = nn.Conv2d(c_top, width, 1)
        self.lat2 = nn.Conv2d(c_skip2, width, 1)
        self.lat1 = nn.Conv2d(c_skip1, width, 1)
        self.mix2 = nn.Sequential(ConvBNAct(width, width, 3, groups=width), ConvBNAct(width, width, 1))
        self.mix1 = nn.Sequential(ConvBNAct(width, width, 3, groups=width), ConvBNAct(width, width, 1))
        self.film = nn.Linear(cond_dim, 2 * width) if cond_dim > 0 else None
        self.score = nn.Conv2d(width, 1, 1)
        self.offset = nn.Conv2d(width, head.sub[0] * head.sub[1], 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, fmap: torch.Tensor, skips: list[torch.Tensor], cond: torch.Tensor | None = None) -> PointerLogits:
        s1, s2 = skips
        x = F.interpolate(self.lat_top(fmap), size=s2.shape[-2:], mode="nearest") + self.lat2(s2)
        x = self.mix2(x)
        x = F.interpolate(x, size=s1.shape[-2:], mode="nearest") + self.lat1(s1)
        if self.film is not None and cond is not None:
            gamma, beta = self.film(cond)[:, :, None, None].chunk(2, dim=1)
            x = x * (1 + gamma) + beta
        x = self.mix1(x)
        score = self.score(x)
        offset = self.offset(x)
        if score.shape[-2:] != tuple(self.grid):
            score = F.interpolate(score, size=self.grid, mode="bilinear", align_corners=False)
            offset = F.interpolate(offset, size=self.grid, mode="bilinear", align_corners=False)
        b = score.shape[0]
        return PointerLogits(score.reshape(b, -1), offset.reshape(b, offset.shape[1], -1), self.grid, self.sub)


class FoveaPool(nn.Module):
    """Foveal attention: a learned saliency map picks *where* to read local evidence (JigokuNet).

    Inspired by spatial-softmax keypoints in visuomotor policies (Levine et al., JMLR 2016): with a
    keyboard, the decisive evidence is the bullet configuration around the player's ship, so the head
    learns (from the BC loss alone) to concentrate on it.
    """

    def __init__(self, c_top: int, c_skip2: int, width: int = 64, glimpses: int = 2):
        super().__init__()
        self.lat_top = nn.Conv2d(c_top, width, 1)
        self.lat2 = nn.Conv2d(c_skip2, width, 1)
        self.mix = nn.Sequential(ConvBNAct(width, width, 3, groups=width), ConvBNAct(width, width, 1))
        self.saliency = nn.Conv2d(width, glimpses, 1)
        self.glimpses = glimpses
        self.out_dim = width * glimpses

    def forward(self, fmap: torch.Tensor, skip2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(self.lat_top(fmap), size=skip2.shape[-2:], mode="nearest") + self.lat2(skip2)
        x = self.mix(x)  # (B, C, h, w)
        b, c, h, w = x.shape
        attn = F.softmax(self.saliency(x).flatten(2), dim=-1)  # (B, G, hw)
        feats = torch.einsum("bgn,bcn->bgc", attn, x.flatten(2)).reshape(b, -1)
        ys = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) / h
        xs = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) / w
        a = attn.view(b, self.glimpses, h, w)
        where = torch.stack([(a.sum(2) * xs).sum(-1), (a.sum(3) * ys).sum(-1)], dim=-1)  # (B, G, 2)
        return feats, where
