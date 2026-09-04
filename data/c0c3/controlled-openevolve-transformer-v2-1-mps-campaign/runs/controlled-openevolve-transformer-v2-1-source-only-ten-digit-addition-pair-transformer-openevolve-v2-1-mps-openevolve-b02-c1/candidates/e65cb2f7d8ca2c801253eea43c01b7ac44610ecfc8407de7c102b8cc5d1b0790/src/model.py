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


class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with one global all-entries offset fixed at zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )

        # Preserve the RNG stream of nn.Embedding's constructor.
        torch.empty(num_embeddings, embedding_dim).normal_()

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        flat = full_weight.reshape(-1)
        with torch.no_grad():
            self.weight.copy_(flat[:-1] - flat[-1])

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class GaugeFixedLMHead(nn.Module):
    """Parameter-free output view of the gauge-fixed tied embedding."""

    def __init__(self, embedding: GaugeFixedEmbedding):
        super().__init__()
        object.__setattr__(self, "embedding", embedding)

        # Preserve the RNG stream of the removed nn.Linear constructor.
        discarded = torch.empty(
            embedding.num_embeddings, embedding.embedding_dim
        )
        nn.init.kaiming_uniform_(discarded, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class GaugeFixedResidualProjection(nn.Module):
    """Residual projection modulo its all-ones output direction."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features - 1, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Preserve the RNG stream of the removed full-width Linear constructor.
        discarded_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        torch.empty(out_features).uniform_(-bound, bound)

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight.copy_(full_weight[:-1] - full_weight[-1:])
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 0, 0, 1))
        return F.linear(x, weight, self.bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedResidualProjection(d_model, d_model)
        # One bias per head is a softmax-invisible common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias
        v = v + self.proj.bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        rel_bias = F.pad(self.rel_bias, (0, 1))
        att = att + rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
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
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2._residual_gauge = True

        # Preserve the RNG stream of the original full-width Linear constructor.
        bound = 1.0 / math.sqrt(d_ff)
        torch.empty(d_ff).uniform_(-bound, bound)

        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = GaugeFixedEmbedding(cfg.vocab_size, cfg.d_model)

        # Preserve the constructor RNG stream of the removed position table.
        torch.empty(cfg.max_seq_len, cfg.d_model - 1).normal_()
        self.drop = nn.Dropout(cfg.dropout)
        self.drop._removed_pos_shape = (cfg.max_seq_len, cfg.d_model - 1)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Parameter-free weight tying with the input embedding.
        self.lm_head = GaugeFixedLMHead(self.token_emb)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        removed_pos_shape = getattr(module, "_removed_pos_shape", None)
        if removed_pos_shape is not None:
            torch.empty(removed_pos_shape).normal_()

        if isinstance(module, GaugeFixedEmbedding):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedLMHead):
            module.embedding.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedResidualProjection):
            module.initialize_from_full_normal()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 1, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:-1] - full_weight[-1:])
            else:
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

        x = self.token_emb(idx)
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
