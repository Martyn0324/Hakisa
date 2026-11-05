"""System 2 — the "prefrontal cortex": a language model that looks at Hakisa's visual tokens, reads the
game state as text (and the partner's instructions), thinks briefly and answers with a PLAN line.

Eyes: Hakisa's own from-scratch visual encoder (System 1's cortex) — 16 pooled tokens of the 1/16
feature map + 1 global token — projected into the LLM embedding space (LLaVA-style MLP projector).
No pre-trained vision model is involved; the LLM (Qwen3) contributes only language/reasoning.

Prompt (Qwen chat format; ``<vision>`` = the projected visual tokens)::

    <|im_start|>system ... <|im_end|>
    <|im_start|>user
    Screen: <vision>
    State: lives 2, bombs 1, danger high, I am at bottom left
    Partner: none
    Previous plan: PLAN: zone=bottom focus=off bomb=hold<|im_end|>
    <|im_start|>assistant
    <think>
    Danger is high and I still have 1 bombs, so I bomb now.
    </think>

    PLAN: zone=bottom_left focus=on bomb=now
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from hakisa.prefrontal.intents import IntentSchema
from hakisa.prefrontal.llm import LoRALinear, inject_lora, load_llm

SYSTEM_PROMPT = (
    "You are Hakisa, an AI playing {title} together with a human partner. You see the screen through your own "
    "eyes, think briefly, then answer with one PLAN line."
)


def make_prompt(title: str, state_text: str, instruction: str | None = None, previous: str | None = None) -> tuple[str, str]:
    pre = f"<|im_start|>system\n{SYSTEM_PROMPT.format(title=title)}<|im_end|>\n<|im_start|>user\nScreen: "
    post = (f"\nState: {state_text}\nPartner: {instruction or 'none'}\nPrevious plan: {previous or 'none'}"
            f"<|im_end|>\n<|im_start|>assistant\n")
    return pre, post


def make_completion(plan_line: str, rationale: str | None = None, think: bool = True) -> str:
    if think and rationale:
        return f"<think>\n{rationale}\n</think>\n\n{plan_line}"
    return f"<think>\n\n</think>\n\n{plan_line}"


@torch.no_grad()
def s1_visual_tokens(policy, obs: torch.Tensor, minimap: torch.Tensor | None = None, grid: int = 4) -> torch.Tensor:
    """(B, 1 + grid*grid [+1], C) tokens from System 1's frozen encoder (null intent)."""
    enc = policy.encoder(obs)
    toks = [enc.pooled[:, None], F.adaptive_avg_pool2d(enc.fmap, grid).flatten(2).transpose(1, 2)]
    if getattr(policy, "minimap_encoder", None) is not None and minimap is not None:
        mm = policy.minimap_encoder(minimap).pooled[:, None]
        c = enc.pooled.shape[-1]
        mm = F.pad(mm, (0, c - mm.shape[-1])) if mm.shape[-1] < c else mm[..., :c]  # same width as the main tokens
        toks.append(mm)
    return torch.cat(toks, dim=1).float()


class Prefrontal(nn.Module):
    def __init__(self, llm, tokenizer, vis_dim: int, schema: IntentSchema, title: str, n_vis: int = 17,
                 spec: str = "tiny-qwen3", train_mode: str = "lora", lora_r: int = 16):
        super().__init__()
        self.llm = llm
        self.tok = tokenizer
        self.schema = schema
        self.title = title
        self.n_vis = n_vis
        self.vis_dim = vis_dim
        self.spec = spec
        self.train_mode = train_mode
        hidden = llm.config.hidden_size
        self.projector = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.vis_pos = nn.Parameter(torch.zeros(1, n_vis, hidden))
        if train_mode == "lora":
            inject_lora(self.llm, r=lora_r, alpha=2.0 * lora_r)
        elif train_mode == "full":
            for p in self.llm.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError("train_mode must be 'lora' or 'full'")

    # -- parameters / persistence -------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state(self) -> dict[str, torch.Tensor]:
        """Only what training changed (projector, positions, LoRA or full LLM weights)."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in keep}

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.adapter_state(), d / "prefrontal.pt")
        meta = {"spec": self.spec, "vis_dim": self.vis_dim, "n_vis": self.n_vis, "title": self.title,
                "schema": self.schema.slots, "train_mode": self.train_mode}
        if self.spec == "tiny-qwen3":  # randomly initialised stand-in: its size is not implied by the name
            meta["llm_config"] = self.llm.config.to_dict()
        (d / "prefrontal.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
             local_files_only: bool = False) -> "Prefrontal":
        d = Path(directory)
        meta = json.loads((d / "prefrontal.json").read_text())
        if meta["spec"] == "tiny-qwen3" and "llm_config" in meta:
            from transformers import Qwen3Config, Qwen3ForCausalLM

            from hakisa.prefrontal.llm import ByteTokenizer

            llm = Qwen3ForCausalLM(Qwen3Config.from_dict(meta["llm_config"])).to(device=device, dtype=dtype)
            tok = ByteTokenizer()
        else:
            llm, tok = load_llm(meta["spec"], device, dtype, local_files_only)
        pf = cls(llm, tok, meta["vis_dim"], IntentSchema(meta["schema"]), meta["title"], meta["n_vis"], meta["spec"], meta["train_mode"])
        pf.load_state_dict(torch.load(d / "prefrontal.pt", map_location="cpu"), strict=False)
        return pf.to(device)

    @contextlib.contextmanager
    def swapped(self, state: dict[str, torch.Tensor]) -> Iterator[None]:
        """Temporarily load another adapter state (e.g. the SL reference for GRPO's KL term)."""
        current = self.adapter_state()
        self.load_state_dict(state, strict=False)
        try:
            yield
        finally:
            self.load_state_dict(current, strict=False)

    # -- sequence construction --------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.vis_pos.device

    @property
    def llm_dtype(self) -> torch.dtype:
        return self.llm.get_input_embeddings().weight.dtype

    def _embed(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.llm.get_input_embeddings()(t)

    def _build(self, vis: torch.Tensor, prompts: list[tuple[str, str]], completions: list[str] | None, left_pad: bool):
        seqs, labels = [], []
        vis_emb = (self.projector(vis.to(self.device).float()) + self.vis_pos).to(self.llm_dtype)
        for i, (pre, post) in enumerate(prompts):
            pre_ids, post_ids = self.tok.encode(pre), self.tok.encode(post)
            parts = [self._embed(pre_ids), vis_emb[i], self._embed(post_ids)]
            lab = [-100] * (len(pre_ids) + self.n_vis + len(post_ids))
            if completions is not None:
                comp_ids = self.tok.encode(completions[i]) + [self.tok.eos_token_id]
                parts.append(self._embed(comp_ids))
                lab += comp_ids
            seqs.append(torch.cat(parts))
            labels.append(lab)
        L = max(s.shape[0] for s in seqs)
        H = seqs[0].shape[1]
        emb = torch.zeros(len(seqs), L, H, dtype=self.llm_dtype, device=self.device)
        mask = torch.zeros(len(seqs), L, dtype=torch.long, device=self.device)
        lab_t = torch.full((len(seqs), L), -100, dtype=torch.long, device=self.device)
        for i, (s, lab) in enumerate(zip(seqs, labels)):
            n = s.shape[0]
            sl = slice(L - n, L) if left_pad else slice(0, n)
            emb[i, sl] = s
            mask[i, sl] = 1
            lab_t[i, sl] = torch.tensor(lab, dtype=torch.long, device=self.device)
        return emb, mask, lab_t

    # -- training signals -------------------------------------------------------------------------
    def token_logprobs(self, vis, prompts, completions) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log-probs of the completions (B, L-1) and the matching mask."""
        emb, mask, labels = self._build(vis, prompts, completions, left_pad=False)
        logits = self.llm(inputs_embeds=emb, attention_mask=mask).logits[:, :-1].float()
        target = labels[:, 1:]
        valid = target != -100
        lp = torch.log_softmax(logits, -1).gather(-1, target.clamp_min(0)[..., None])[..., 0]
        return lp * valid, valid

    def loss(self, vis, prompts, completions) -> torch.Tensor:
        lp, valid = self.token_logprobs(vis, prompts, completions)
        return -(lp.sum() / valid.sum().clamp_min(1))

    def sequence_logprob(self, vis, prompts, completions) -> torch.Tensor:
        lp, valid = self.token_logprobs(vis, prompts, completions)
        return lp.sum(-1) / valid.sum(-1).clamp_min(1)

    # -- inference --------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, vis, prompts, max_new_tokens: int = 64, temperature: float = 0.0, top_p: float = 1.0) -> list[str]:
        emb, mask, _ = self._build(vis, prompts, None, left_pad=True)
        kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            kwargs.update(do_sample=False)
        out = self.llm.generate(inputs_embeds=emb, attention_mask=mask, **kwargs)
        texts = []
        for row in out:
            ids = [int(i) for i in row if int(i) not in (self.tok.pad_token_id, self.tok.eos_token_id)]
            texts.append(self.tok.decode(ids, skip_special_tokens=True))
        return texts

    def plan(self, vis, prompts, **kw) -> list[tuple[str, dict[str, str] | None]]:
        texts = self.generate(vis, prompts, **kw)
        return [(t, self.schema.parse_plan(t)) for t in texts]


def count_lora(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, LoRALinear))
