"""A small GPT (nanoGPT-style): pre-norm blocks, causal self-attention, GELU MLP, tied embeddings.

Initialisation is deterministic for a given seed. Every weight matrix dimension is a multiple of 64
when n_embd and the (padded) vocabulary are, so DeMo's 64 x 64 chunks and the 1-D 64- and
256-point chunks tile every tensor exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 128          # padded to a multiple of 64
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.out = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=False)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.dim() == 2:
                    std = 0.02 / math.sqrt(2 * cfg.n_layer) if name.endswith(("proj.weight", "out.weight")) else 0.02
                    p.copy_(torch.randn(p.shape, generator=g) * std)

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.ln_f(x) @ self.wte.weight.T
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
