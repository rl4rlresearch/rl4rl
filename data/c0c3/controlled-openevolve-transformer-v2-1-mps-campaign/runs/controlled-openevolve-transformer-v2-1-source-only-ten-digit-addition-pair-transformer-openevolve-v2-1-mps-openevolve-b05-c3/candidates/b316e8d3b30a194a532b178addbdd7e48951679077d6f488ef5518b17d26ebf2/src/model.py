"""Tiny decoder-only transformer used for 10-digit addition."""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class GaugeFixedEmbedding(nn.Embedding):
    """Embedding vectors represented modulo a shared channel shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

    def full_weight(self) -> torch.Tensor:
        zero = self.weight.new_zeros(self.num_embeddings, 1)
        return torch.cat((self.weight, zero), dim=-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class GaugeFixedKeyLinear(nn.Linear):
    """QKV projection with one key-row coefficient fixed by LayerNorm gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.missing_index = d_model * d_model + d_model - 1
        self.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and four query biases remain.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
        qkv = F.linear(x, self.qkv.full_weight(), bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
        elif isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[d_model, -1].clone()
                full[d_model, :-1].sub_(omitted)
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_index],
                            flat[module.missing_index + 1 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, seqlen = idx.shape

        if seqlen > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.cfg.max_seq_len :]
            seqlen = idx.shape[1]

        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        out = prompt
        for _ in range(max_new_tokens):
            idx = out[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(idx)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
