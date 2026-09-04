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
        # Key bias is softmax-invariant, and every value-bias coordinate can
        # be absorbed by the downstream projection bias. Store only query
        # bias and reconstruct the other two bias vectors in fixed gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
        # The feature-uniform component of this residual bias is canceled by
        # downstream LayerNorms, so retain only its relative coordinates.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        query_bias = self.qkv.bias[:d_model]
        value_bias = self.qkv.bias.new_zeros(d_model)
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )
        qkv_weight_relative = torch.cat(
            (
                self.qkv.weight,
                self.qkv.weight.new_zeros(
                    (self.qkv.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        qkv_weight = (
            qkv_weight_relative
            + self.qkv.weight.mean(dim=-1, keepdim=True)
        )
        qkv = F.linear(x, qkv_weight, qkv_bias)
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
        weight_relative = torch.cat(
            (
                self.proj.weight,
                self.proj.weight.new_zeros(
                    (self.proj.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        proj_weight = (
            weight_relative
            + self.proj.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # The final LayerNorm cancels the feature-uniform component of this
        # residual bias, so retain only its relative coordinates.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight_relative = torch.cat(
            (
                self.fc1.weight,
                self.fc1.weight.new_zeros(
                    (self.fc1.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc1_weight = (
            fc1_weight_relative
            + self.fc1.weight.mean(dim=-1, keepdim=True)
        )
        hidden = F.gelu(F.linear(x, fc1_weight, self.fc1.bias))
        weight_relative = torch.cat(
            (
                self.fc2.weight,
                self.fc2.weight.new_zeros(
                    (self.fc2.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc2_weight = (
            weight_relative
            + self.fc2.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, fc2_weight, fc2_bias))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Learned scales are folded into the downstream weights by the
        # factor-aware optimizer.
        self.ln1 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
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

        # LayerNorm-null input coordinates and final-LayerNorm-null output
        # coordinates are retained only through relative representatives.
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach()
            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1)
            )
            block.attn.proj.weight = nn.Parameter(
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            full_fc1_weight = block.mlp.fc1.weight.detach()
            block.mlp.fc1.weight = nn.Parameter(
                full_fc1_weight[:, :-1] - full_fc1_weight[:, -1:]
            )
            full_fc2_weight = (
                block.mlp.fc2.weight.detach().transpose(0, 1)
            )
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )

        # A global feature-uniform shift of every tied token row is canceled
        # at the input and becomes a vocabulary-uniform output-logit shift.
        full_token_weight = self.token_emb.weight.detach().flatten()
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - full_token_weight[-1]
        )
        self.lm_head.weight = self.token_emb.weight

        # Uniform feature offsets in positional rows are canceled by all
        # downstream LayerNorms. Store only relative coordinates per row.
        full_pos_weight = self.pos_emb.weight.detach()
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
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

        token_relative = torch.cat(
            (
                self.token_emb.weight,
                self.token_emb.weight.new_zeros(1),
            )
        )
        token_weight = (
            token_relative + self.token_emb.weight.mean()
        ).view(self.cfg.vocab_size, self.cfg.d_model)

        pos_relative = torch.cat(
            (
                self.pos_emb.weight,
                self.pos_emb.weight.new_zeros(
                    (self.pos_emb.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        pos_weight = pos_relative + self.pos_emb.weight.mean(
            dim=-1, keepdim=True
        )
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = F.linear(x, token_weight)

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
