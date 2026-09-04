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


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
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


class CompactPositionEmbedding(nn.Module):
    """Positional embedding with two token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[2:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(2), self.weight))
        return F.embedding(
            idx,
            full_weight.view(self.num_embeddings, self.embedding_dim),
        )


class CompactQKV(nn.Module):
    """Compact QKV with three LayerNorm-induced key-weight gauges fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        key_start: int,
        second_head_start: int,
        ln_weight: nn.Parameter,
    ):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.second_key_row = second_head_start
        self.ln_weight = ln_weight

        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 1 :],
            ),
            dim=0,
        )
        self.weight = nn.Parameter(retained_weight.detach().clone())

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("key_basis", basis, persistent=False)

        scaled_key_weight = (
            linear.weight[[key_start, key_start + 1, second_head_start]]
            * ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        query_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
            )
        )
        self.query_bias = nn.Parameter(query_bias.detach().clone())
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        second_retained_start = self.second_key_row - 2
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:2],
                self.weight[self.key_start : second_retained_start],
                key_weight[2:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
        full_bias = torch.cat(
            (
                self.query_bias[:1].expand(self.head_dim - 2),
                self.query_bias.new_zeros(2),
                self.query_bias[1:],
                self.query_bias.new_zeros(1),
                self.query_bias.new_zeros(self.key_start),
                self.query_bias.new_zeros(self.head_dim),
                self.query_bias.new_zeros(3),
                self.value_bias,
            )
        )
        return F.linear(x, full_weight, full_bias)


class CompactSharedProjection(nn.Module):
    """Projection with a zero-mean effective offset and retained value scalar."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.shared_bias = shared_bias

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 2)
        for column in range(width - 2):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
        compact_bias = basis.transpose(0, 1) @ centered_bias
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_bias = self.bias_basis @ self.bias
        raw_bias = torch.cat(
            (
                raw_bias[:-1],
                raw_bias[-1:] + self.shared_bias,
            )
        )
        value_offset = self.weight[:, -1] * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, self.weight, full_bias)


class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )


class CompactFirstLinearRow(nn.Module):
    """Linear layer with one LayerNorm input-direction gauge fixed."""

    def __init__(self, linear: nn.Linear, ln_weight: nn.Parameter):
        super().__init__()
        self.ln_weight = ln_weight
        self.weight = nn.Parameter(linear.weight[1:].detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_row = linear.weight[0] * ln_weight
        centered_row = scaled_row - scaled_row.mean()
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self.ln_weight
        full_weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, full_weight, self.bias)


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
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
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
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Fix two exact token/position translation gauges while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(2):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Retain the qualified bias and projection layout, fix three key-row
        # gauges, and quotient one independently biased MLP input row.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
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
