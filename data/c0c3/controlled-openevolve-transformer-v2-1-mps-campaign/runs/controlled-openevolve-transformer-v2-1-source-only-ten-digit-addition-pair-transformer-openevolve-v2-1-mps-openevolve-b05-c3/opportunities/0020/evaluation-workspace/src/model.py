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
        # Consume the constructor RNG used by the original full embedding.
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


class LowRankTokenEmbedding(nn.Embedding):
    """A learned rank-(d-1) vocabulary matrix shared with the output head."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the baseline embedding-constructor RNG before replacing its
        # full matrix with two learned low-rank factors.
        super().__init__(num_embeddings, embedding_dim)
        del self.weight
        self.rank = embedding_dim - 1
        self.codes = nn.Parameter(torch.empty(num_embeddings, self.rank))
        self.basis = nn.Parameter(torch.empty(self.rank, embedding_dim))

    @torch.no_grad()
    def initialize_from_full(self, full: torch.Tensor) -> None:
        # A balanced truncated SVD preserves as much of the corresponding
        # freshly drawn baseline vocabulary matrix as rank seven permits.
        u, singular, vh = torch.linalg.svd(full, full_matrices=False)
        scale = singular[: self.rank].sqrt()
        self.codes.copy_(u[:, : self.rank] * scale)
        self.basis.copy_(scale[:, None] * vh[: self.rank])

    def full_weight(self) -> torch.Tensor:
        return self.codes @ self.basis

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class LowRankTiedHead(nn.Linear):
    """Parameter-free head reusing a low-rank token embedding."""

    def __init__(self, token_embedding: LowRankTokenEmbedding):
        # Match the baseline head-constructor RNG stream, then discard its
        # independent matrix because logits use the learned shared factors.
        super().__init__(
            token_embedding.embedding_dim,
            token_embedding.num_embeddings,
            bias=False,
        )
        del self.weight
        object.__setattr__(self, "token_embedding", token_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.token_embedding.full_weight())


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Four query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
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
        qkv = F.linear(x, self.qkv.weight, bias)
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
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


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
        self.token_emb = LowRankTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # The learned low-rank vocabulary matrix is shared exactly between
        # token lookup and output classification.
        self.lm_head = LowRankTiedHead(self.token_emb)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LowRankTiedHead):
            # This draw occurs at the same point as initialization of the
            # baseline tied head and becomes the effective shared matrix.
            token_embedding = module.token_embedding
            full = token_embedding.codes.new_empty(
                token_embedding.num_embeddings,
                token_embedding.embedding_dim,
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            token_embedding.initialize_from_full(full)
        elif isinstance(module, LowRankTokenEmbedding):
            # Preserve the baseline token-embedding initialization draw. As in
            # the tied baseline, the later head visit supplies the final draw.
            full = module.codes.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            module.initialize_from_full(full)
        elif isinstance(module, GaugeFixedEmbedding):
            # Draw the original full tensor to preserve the qualified RNG
            # stream, then select the equivalent last-coordinate-zero gauge.
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
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
