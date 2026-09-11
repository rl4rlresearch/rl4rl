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
    """Tied embedding with each token's scalar row offset fixed at zero."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 1)
        )

        # Preserve the RNG stream of nn.Embedding's constructor.
        torch.empty(num_embeddings, embedding_dim).normal_()

    def initialize_from_full_normal(self) -> None:
        full_weight = self.weight.new_empty(
            self.num_embeddings, self.embedding_dim
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight.copy_(
                full_weight[:, :-1] - full_weight[:, -1:]
            )

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1))

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
        # Only learned within-coordinate contrasts are observable by the
        # row-gauge-fixed tied classifier.
        x = x - x.mean(dim=-1, keepdim=True)
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
        self.bias = nn.Parameter(torch.zeros(out_features - 1))

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
        bias = F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)


class GaugeFixedQKV(nn.Module):
    """Basis-gauge-fixed Q/K and value factors."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.score_dim = self.head_dim - 2
        reduced_width = d_model - 1
        self.query_weight = nn.Parameter(
            torch.empty(n_head, self.score_dim, reduced_width)
        )
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.score_dim,
                reduced_width - self.score_dim,
            )
        )
        self.query_bias = nn.Parameter(
            torch.empty(n_head, self.score_dim)
        )
        self.value_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.head_dim,
                reduced_width - self.head_dim,
            )
        )

        # Preserve the RNG stream of the removed bias-free QKV Linear.
        discarded_weight = torch.empty(3 * d_model, d_model)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    def initialize_from_full_normal(self) -> None:
        full_weight = self.value_tail.new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        full_query_weight = full_weight[: self.d_model]
        full_key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
        value_weight = full_weight[2 * self.d_model :]
        query_weight = (
            full_query_weight[:, :-1] - full_query_weight[:, -1:]
        )
        key_weight = (
            full_key_weight[:, :-1] - full_key_weight[:, -1:]
        )
        effective_value_weight = (
            value_weight[:, :-1] - value_weight[:, -1:]
        )
        query_bias = full_weight.new_zeros(self.d_model)
        full_query_heads = query_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        full_key_heads = key_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        query_heads = full_query_heads[:, : self.score_dim]
        key_heads = full_key_heads[:, : self.score_dim]
        value_heads = effective_value_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        full_bias_heads = query_bias.view(
            self.n_head, self.head_dim
        )
        bias_heads = full_bias_heads[:, : self.score_dim]

        with torch.no_grad():
            key_basis = key_heads[..., : self.score_dim]
            self.query_weight.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            self.key_tail.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., self.score_dim :]
                )
            )
            self.query_bias.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
            value_basis = value_heads[..., : self.head_dim]
            self.value_tail.copy_(
                torch.linalg.solve(
                    value_basis, value_heads[..., self.head_dim :]
                )
            )

        # Consumed by QuotientAdamW to preserve virtual rank-three updates.
        self._initial_query_weight = (
            query_heads.reshape(-1, self.d_model - 1).detach().clone()
        )
        self._initial_key_weight = (
            key_heads.reshape(-1, self.d_model - 1).detach().clone()
        )
        self._initial_query_bias = (
            bias_heads.reshape(-1).detach().clone()
        )
        self._initial_value_weight = value_weight.detach().clone()

    def forward(
        self, normalized_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reduced_x = normalized_x[..., :-1]
        key_prefix = torch.eye(
            self.score_dim,
            device=reduced_x.device,
            dtype=reduced_x.dtype,
        ).expand(self.n_head, -1, -1)
        key_weight = torch.cat([key_prefix, self.key_tail], dim=-1)
        q = torch.einsum(
            "btf,hdf->bhtd", reduced_x, self.query_weight
        )
        q = q + self.query_bias.unsqueeze(0).unsqueeze(2)
        k = torch.einsum(
            "bsf,hdf->bhsd", reduced_x, key_weight
        )
        att = torch.einsum("bhtd,bhsd->bhts", q, k)
        value_prefix = torch.eye(
            self.head_dim,
            device=reduced_x.device,
            dtype=reduced_x.dtype,
        ).expand(self.n_head, -1, -1)
        value_weight = torch.cat(
            [value_prefix, self.value_tail], dim=-1
        ).reshape(self.d_model, self.d_model - 1)
        v = F.linear(reduced_x, value_weight)
        return att, v


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.proj = GaugeFixedResidualProjection(d_model, d_model)
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def initialize_value_basis(self) -> None:
        full_value_weight = self.qkv._initial_value_weight
        effective_value_weight = (
            full_value_weight[:, :-1] - full_value_weight[:, -1:]
        )
        value_heads = effective_value_weight.view(
            self.n_head, self.head_dim, -1
        )
        value_basis = value_heads[..., : self.head_dim]
        self.qkv.value_tail.copy_(
            torch.linalg.solve(
                value_basis, value_heads[..., self.head_dim :]
            )
        )

        virtual_proj_weight = self.proj.weight.detach().clone()
        self.qkv._initial_proj_weight = virtual_proj_weight
        proj_heads = virtual_proj_weight.view(
            self.proj.out_features - 1,
            self.n_head,
            self.head_dim,
        )
        canonical_proj = torch.einsum(
            "ohd,hde->ohe", proj_heads, value_basis
        )
        self.proj.weight.copy_(
            canonical_proj.reshape_as(self.proj.weight)
        )

    def forward(
        self, x: torch.Tensor, normalized_x: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        att, v = self.qkv(normalized_x)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = att / math.sqrt(self.qkv.score_dim)
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        rel_bias = torch.stack(
            [
                F.pad(self.rel_bias[0], (0, 2)),
                torch.cat(
                    [
                        self.rel_bias[1][:-1],
                        0.75 * self.rel_bias[1][-2:-1]
                        + 0.25 * self.rel_bias[1][-1:],
                        0.375 * self.rel_bias[1][-2:-1]
                        + 0.625 * self.rel_bias[1][-1:],
                        self.rel_bias[1][-1:].expand(4),
                        self.rel_bias[1].new_zeros(3),
                    ]
                ),
            ]
        )
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

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        return self.drop(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            x,
            self.ln1.normalized_shape,
            weight=None,
            bias=None,
            eps=self.ln1.eps,
        )
        x = x + self.attn(normalized, normalized)
        x = x + self.mlp(self.ln2(x))
        return x


class GaugeFixedFinalLayerNorm(nn.Module):
    """Final LayerNorm modulo common bias and residual-scale gauges."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            [
                self.weight,
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
        bias = torch.cat(
            [
                self.bias,
                self.bias[-1:],
                self.bias.new_zeros(1),
            ]
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            bias,
            self.eps,
        )


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
        self.ln_f = GaugeFixedFinalLayerNorm(cfg.d_model)

        # Parameter-free weight tying with the input embedding.
        self.lm_head = GaugeFixedLMHead(self.token_emb)

        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.initialize_value_basis()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        removed_pos_shape = getattr(module, "_removed_pos_shape", None)
        if removed_pos_shape is not None:
            torch.empty(removed_pos_shape).normal_()

        if isinstance(module, GaugeFixedQKV):
            module.initialize_from_full_normal()
        elif isinstance(module, GaugeFixedEmbedding):
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
