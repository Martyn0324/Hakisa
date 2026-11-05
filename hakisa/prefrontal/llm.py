"""Language-model plumbing for System 2: loading Qwen3 (or a tiny stand-in), LoRA, a byte tokenizer.

``load_llm("Qwen/Qwen3-0.6B")`` loads the real pre-trained model from the Hugging Face Hub (text
only — allowed by the project rules).  ``load_llm("tiny-qwen3")`` builds a *randomly initialised*
two-layer model with the same Qwen3 architecture and a byte-level tokenizer: it is only used to test
the code paths where the Hub is unreachable; it has no pre-trained knowledge.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class ByteTokenizer:
    """Minimal tokenizer (UTF-8 bytes + 3 specials) with the subset of the HF API Hakisa uses."""

    def __init__(self):
        self.pad_token_id, self.eos_token_id, self.bos_token_id = 256, 257, 258
        self.vocab_size = 259

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        data = bytes(int(i) for i in ids if int(i) < 256)
        return data.decode("utf-8", errors="ignore")


class LoRALinear(nn.Module):
    """y = W x + (B A x) * alpha / r, with W frozen (Hu et al., 2021)."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lora_a.to(x.dtype)
        b = self.lora_b.to(x.dtype)
        return self.base(x) + (self.dropout(x) @ a.t() @ b.t()) * self.scale


DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.0,
                targets: tuple[str, ...] = DEFAULT_LORA_TARGETS) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    params: list[nn.Parameter] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                lora = LoRALinear(child, r, alpha, dropout)
                setattr(module, child_name, lora)
                params += [lora.lora_a, lora.lora_b]
    if not params:
        raise ValueError("no LoRA target modules found")
    return params


def tiny_qwen3(vocab_size: int = 259, hidden: int = 128, layers: int = 2, heads: int = 4):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=max(1, heads // 2), head_dim=hidden // heads,
        max_position_embeddings=2048, tie_word_embeddings=True, pad_token_id=256, eos_token_id=257, bos_token_id=258,
    )
    return Qwen3ForCausalLM(cfg)


def load_llm(spec: str, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32, local_files_only: bool = False):
    """Returns (model, tokenizer). ``spec`` = HF model id/path, or ``tiny-qwen3`` for tests."""
    if spec == "tiny-qwen3":
        tok = ByteTokenizer()
        return tiny_qwen3(tok.vocab_size).to(device=device, dtype=dtype), tok
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec, local_files_only=local_files_only)
    try:
        model = AutoModelForCausalLM.from_pretrained(spec, dtype=dtype, local_files_only=local_files_only)
    except TypeError:  # older transformers spell it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(spec, torch_dtype=dtype, local_files_only=local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model.to(device), tok
