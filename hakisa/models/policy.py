"""Hakisa's policy network: cortex encoder -> trunk -> basal-ganglia heads (+ critic, + amygdala).

The generic :class:`HakisaPolicy` is composed differently per game (see ``hakisa.models.games``):

* ``fovea``     foveal attention pooling (keyboard games: read evidence around the ship)
* pointer decoders for every pointer head (mouse games: superior-colliculus target map)
* ``minimap``   second encoder stream (LoL)
* ``aux``       non-visual game-state vector (LoL live-client stats)
* intents       System-2 plan slots embedded and injected with FiLM (null intent = index 0 everywhere)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from hakisa.basal_ganglia.heads import ActionDistribution, FoveaPool, PointerDecoder, PolicyOutput
from hakisa.config import GameConfig
from hakisa.cortex.blocks import mlp
from hakisa.cortex.encoder import EncoderOutput, VisualEncoder


@dataclass
class Features:
    enc: EncoderOutput
    trunk: torch.Tensor
    minimap: EncoderOutput | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


def intent_sizes_from_config(cfg: GameConfig) -> list[int]:
    slots = (cfg.intents or {}).get("slots") or {}
    return [len(v) for v in slots.values()]


class HakisaPolicy(nn.Module):
    arch = "generic"
    use_fovea = False

    def __init__(self, cfg: GameConfig, hidden: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.space = cfg.action_space
        m = cfg.model
        preset = m.get("preset", "small")
        hidden = int(hidden or m.get("hidden", 256 if preset != "nano" else 160))
        self.hidden = hidden
        self.intent_sizes = intent_sizes_from_config(cfg)
        intent_dim = int(m.get("intent_dim", 32)) if self.intent_sizes else 0
        self.intent_dim = intent_dim
        self.intent_emb = nn.ModuleList([nn.Embedding(n, intent_dim) for n in self.intent_sizes])
        for emb in self.intent_emb:  # index 0 ("any") = null intent -> start at zero
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()

        obs = cfg.obs
        self.encoder = VisualEncoder(obs.channels, (obs.height, obs.width), preset=preset, film_dim=intent_dim)
        c3 = self.encoder.out_dim
        c1, c2 = self.encoder.skip_dims
        parts = c3
        self.fovea = None
        if self.use_fovea:
            self.fovea = FoveaPool(c3, c2, width=int(m.get("fovea_width", 64 if preset != "nano" else 48)))
            parts += self.fovea.out_dim
        self.minimap_encoder = None
        if obs.minimap is not None:
            self.minimap_encoder = VisualEncoder(
                3 * obs.stack, (obs.minimap.height, obs.minimap.width), preset=m.get("minimap_preset", "nano")
            )
            parts += self.minimap_encoder.out_dim
        self.aux_mlp = None
        if obs.aux_dim > 0:
            self.aux_mlp = mlp(obs.aux_dim, 64, 64)
            parts += 64
        parts += intent_dim
        self.trunk = mlp(parts, hidden, hidden, layers=2)
        self.trunk_act = nn.GELU()

        self.cat_heads = nn.ModuleDict({h.name: nn.Linear(hidden, h.num_classes) for h in self.space.categorical_heads})
        for lin in self.cat_heads.values():
            nn.init.normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)
        self.cond_emb = nn.ModuleDict()
        self.pointer_decoders = nn.ModuleDict()
        width = int(m.get("pointer_width", 48 if preset != "nano" else 32))
        for p in self.space.pointer_heads:
            cond_dim = hidden
            if p.conditioned_on:
                k = self.space.head(p.conditioned_on).num_classes
                self.cond_emb[p.name] = nn.Embedding(k, 32)
                cond_dim += 32
            if p.source == "minimap":
                assert self.minimap_encoder is not None
                enc = self.minimap_encoder
            else:
                enc = self.encoder
            e1, e2 = enc.skip_dims
            self.pointer_decoders[p.name] = PointerDecoder(enc.out_dim, e2, e1, p, width=width, cond_dim=cond_dim)
        self.value_head = mlp(hidden, hidden // 2, 1)
        self.danger_head = nn.Linear(hidden, 1)

    # -- helpers ----------------------------------------------------------------------------------
    def embed_intent(self, intent: torch.Tensor | None, batch: int, device) -> torch.Tensor | None:
        if not self.intent_sizes:
            return None
        if intent is None:
            intent = torch.zeros(batch, len(self.intent_sizes), dtype=torch.long, device=device)
        return sum(emb(intent[:, i]) for i, emb in enumerate(self.intent_emb))  # type: ignore[return-value]

    def freeze_bn(self) -> None:
        """Use running BN statistics (call for RL: rollout and update then see the same network)."""
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()

    def train(self, mode: bool = True):  # keep BN frozen across .train() calls once requested
        super().train(mode)
        if mode and getattr(self, "_bn_frozen", False):
            self.freeze_bn()
        return self

    def set_bn_frozen(self, frozen: bool = True) -> None:
        self._bn_frozen = frozen
        if frozen:
            self.freeze_bn()

    # -- forward ----------------------------------------------------------------------------------
    def features(
        self,
        obs: torch.Tensor,
        aux: torch.Tensor | None = None,
        minimap: torch.Tensor | None = None,
        intent: torch.Tensor | None = None,
    ) -> Features:
        b = obs.shape[0]
        z = self.embed_intent(intent, b, obs.device)
        enc = self.encoder(obs, film=z)
        parts = [enc.pooled]
        extras: dict[str, torch.Tensor] = {}
        if self.fovea is not None:
            feats, where = self.fovea(enc.fmap, enc.skips[1])
            parts.append(feats)
            extras["fovea_xy"] = where
        mm = None
        if self.minimap_encoder is not None:
            if minimap is None:
                raise ValueError("this policy needs the minimap stream")
            mm = self.minimap_encoder(minimap)
            parts.append(mm.pooled)
        if self.aux_mlp is not None:
            if aux is None:
                aux = torch.zeros(b, self.cfg.obs.aux_dim, device=obs.device)
            parts.append(self.aux_mlp(aux.float()))
        if z is not None:
            parts.append(z)
        trunk = self.trunk_act(self.trunk(torch.cat(parts, dim=-1)))
        return Features(enc=enc, trunk=trunk, minimap=mm, extras=extras)

    def heads(
        self,
        feats: Features,
        condition: dict[str, torch.Tensor] | None = None,
        categorical: bool = True,
        pointers: bool = True,
    ) -> PolicyOutput:
        out = PolicyOutput(extras=dict(feats.extras))
        if categorical:
            out.logits = {name: head(feats.trunk) for name, head in self.cat_heads.items()}
            out.value = self.value_head(feats.trunk)[:, 0]
            out.danger = self.danger_head(feats.trunk)[:, 0]
        if pointers:
            for p in self.space.pointer_heads:
                cond = feats.trunk
                if p.conditioned_on:
                    if condition is not None and p.conditioned_on in condition:
                        c = condition[p.conditioned_on].clamp_min(0)
                    else:  # no conditioning given: use the most likely command
                        c = self.cat_heads[p.conditioned_on](feats.trunk).argmax(-1)
                    cond = torch.cat([cond, self.cond_emb[p.name](c)], dim=-1)
                enc = feats.minimap if p.source == "minimap" else feats.enc
                assert enc is not None
                out.pointer[p.name] = self.pointer_decoders[p.name](enc.fmap, enc.skips, cond)
        return out

    def forward(
        self,
        obs: torch.Tensor,
        aux: torch.Tensor | None = None,
        minimap: torch.Tensor | None = None,
        intent: torch.Tensor | None = None,
        condition: dict[str, torch.Tensor] | None = None,
    ) -> PolicyOutput:
        return self.heads(self.features(obs, aux, minimap, intent), condition)

    def distribution(self, out: PolicyOutput) -> ActionDistribution:
        return ActionDistribution(self.space, out)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        aux: torch.Tensor | None = None,
        minimap: torch.Tensor | None = None,
        intent: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, PolicyOutput]:
        feats = self.features(obs, aux, minimap, intent)
        out = self.heads(feats, pointers=False)
        dist = self.distribution(out)
        actions = dist.sample_categorical(deterministic)
        ptr = self.heads(feats, condition=actions, categorical=False)
        out.pointer = ptr.pointer
        actions.update(dist.sample_pointers(deterministic))
        log_prob = dist.log_prob(actions)
        assert out.value is not None
        return actions, log_prob, out.value, out

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: dict[str, torch.Tensor],
        aux: torch.Tensor | None = None,
        minimap: torch.Tensor | None = None,
        intent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ActionDistribution]:
        out = self.forward(obs, aux, minimap, intent, condition=actions)
        dist = self.distribution(out)
        assert out.value is not None
        return dist.log_prob(actions), dist.entropy(actions), out.value, dist

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
