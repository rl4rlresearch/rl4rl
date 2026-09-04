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


class ScaleFixedLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)


class ShiftGaugeBias(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.coordinates = nn.Parameter(torch.zeros(size - 1))

        inv_sqrt = size ** -0.5
        reflector = torch.full((size,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

    def forward(self) -> torch.Tensor:
        bias = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, bias) / self.reflector_norm_sq
        return bias - self.reflector * projection


class OneColumnGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))
        self.rest_weight = nn.Parameter(torch.empty(out_features, in_features - 4))
        self.middle_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.last_coordinates = nn.Parameter(torch.empty(out_features - 1))

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Match the original Linear(10, 8) constructor's random draw so the
        # already-successful leading coordinates retain their initialization.
        conceptual_rest = torch.empty(out_features, in_features - 2)
        nn.init.kaiming_uniform_(conceptual_rest, a=math.sqrt(5))
        nn.init.normal_(self.first_coordinates, mean=0.0, std=0.02)
        self._set_rest(conceptual_rest)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * (value @ self.reflector) / self.reflector_norm_sq
        return value - projection.unsqueeze(-1) * self.reflector

    def _set_rest(self, conceptual_rest: torch.Tensor) -> None:
        with torch.no_grad():
            self.rest_weight.copy_(conceptual_rest[:, :-2])
            transformed_middle = self._householder(conceptual_rest[:, -2])
            self.middle_coordinates.copy_(transformed_middle[1:])
            transformed_last = self._householder(conceptual_rest[:, -1])
            self.last_coordinates.copy_(transformed_last[1:])

    def reset_rest_parameters(self) -> None:
        conceptual_rest = self.rest_weight.new_empty(
            self.out_features, self.in_features - 2
        )
        nn.init.normal_(conceptual_rest, mean=0.0, std=0.02)
        self._set_rest(conceptual_rest)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        first = self._householder(first)
        middle = F.pad(self.middle_coordinates, (1, 0))
        middle = self._householder(middle)
        last = F.pad(self.last_coordinates, (1, 0))
        last = self._householder(last)
        weight = torch.cat(
            (
                first.transpose(0, 1),
                self.rest_weight,
                middle.unsqueeze(1),
                last.unsqueeze(1),
            ),
            dim=1,
        )
        return F.linear(x, weight)


class ShiftGaugeEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        size = num_embeddings * embedding_dim
        self.coordinates = nn.Parameter(torch.empty(size - 1))

        inv_sqrt = size ** -0.5
        reflector = torch.full((size,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coordinates, mean=0.0, std=0.02)

    def materialized_weight(self) -> torch.Tensor:
        flat = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, flat) / self.reflector_norm_sq
        flat = flat - self.reflector * projection
        return flat.view(self.num_embeddings, self.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 6))
        v = v + F.pad(self.v_bias, (0, 5))

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
        y = self.proj(y) + F.pad(self.proj_bias, (0, 1))
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = OneColumnGaugeLinear(d_ff, d_model)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))) + self.output_bias())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = ScaleFixedLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
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
        logits = F.linear(x, self.token_emb.materialized_weight())

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
