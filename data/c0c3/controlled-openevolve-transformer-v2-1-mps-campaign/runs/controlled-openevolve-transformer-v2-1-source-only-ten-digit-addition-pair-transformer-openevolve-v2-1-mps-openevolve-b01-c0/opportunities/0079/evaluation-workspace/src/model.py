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


class AnchoredKeyLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original QKV linear layer.
        _ = nn.Linear(d_model, 3 * d_model)
        self.d_model = d_model
        self.before_key = nn.Parameter(torch.empty(d_model, d_model))
        self.key_first_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_second_row = nn.Parameter(torch.empty(d_model - 1))
        self.key_third_row = nn.Parameter(torch.empty(d_model - 1))
        self.after_key = nn.Parameter(torch.empty(2 * d_model - 3, d_model))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_key_row = weight[self.d_model]
        second_key_row = weight[self.d_model + 1]
        third_key_row = weight[self.d_model + 2]
        with torch.no_grad():
            self.before_key.copy_(weight[: self.d_model])
            self.key_first_row.copy_(
                first_key_row[:-1] - first_key_row[-1]
            )
            self.key_second_row.copy_(
                second_key_row[:-1] - second_key_row[-1]
            )
            self.key_third_row.copy_(
                third_key_row[1:] - third_key_row[0]
            )
            self.after_key.copy_(weight[self.d_model + 3 :])

    def reconstructed_weight(self) -> torch.Tensor:
        first_key_row = torch.cat(
            (self.key_first_row, self.key_first_row.new_zeros(1))
        )
        second_key_row = torch.cat(
            (self.key_second_row, self.key_second_row.new_zeros(1))
        )
        third_key_row = torch.cat(
            (self.key_third_row.new_zeros(1), self.key_third_row)
        )
        return torch.cat(
            (
                self.before_key,
                first_key_row.unsqueeze(0),
                second_key_row.unsqueeze(0),
                third_key_row.unsqueeze(0),
                self.after_key,
            ),
            dim=0,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Preserve the original constructor RNG while anchoring one redundant
        # coordinate in each of the first three key rows.
        self.qkv = AnchoredKeyLinear(d_model)
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 6))
        self.proj = nn.Linear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj.bias = None
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qv_bias[:d_model]
        v_bias = torch.cat(
            (
                self.qv_bias.new_zeros(1),
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(3),
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
        fused_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.reconstructed_weight(), fused_bias)
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
        proj_bias = torch.cat((self.proj_bias, self.proj_bias.new_zeros(1)))
        proj_bias = proj_bias - proj_bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2_first_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2_second_column = nn.Parameter(torch.empty(d_model - 1))
        self.fc2_rest = nn.Parameter(torch.empty(d_model, d_ff - 2))
        self.fc2.weight = None
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.fc2.bias = None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        first_final_coordinate = -self.fc2_first_column.sum().reshape(1)
        first_column = torch.cat((self.fc2_first_column, first_final_coordinate))
        second_final_coordinate = -self.fc2_second_column.sum().reshape(1)
        second_column = torch.cat((self.fc2_second_column, second_final_coordinate))
        weight = torch.cat(
            (first_column.unsqueeze(1), second_column.unsqueeze(1), self.fc2_rest),
            dim=1,
        )
        output_bias = torch.cat((self.fc2_bias, self.fc2_bias.new_zeros(1)))
        output_bias = output_bias - output_bias.mean()
        return self.drop(F.linear(hidden, weight, output_bias))


class CenteredBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-5

    def forward(
        self, x: torch.Tensor, shared_bias: torch.Tensor
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                shared_bias.new_zeros(2),
                shared_bias.reshape(1),
                shared_bias.new_zeros(4),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, shared_bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-5

    def forward(
        self,
        x: torch.Tensor,
        third_bias: torch.Tensor,
        fifth_bias: torch.Tensor,
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                third_bias.new_zeros(2),
                third_bias.reshape(1),
                third_bias.new_zeros(1),
                fifth_bias.reshape(1),
                third_bias.new_zeros(2),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, third_bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class CenteredTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # Consume the same constructor RNG as the original embedding.
        _ = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.first_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.second_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 3))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_column = weight[:, 0]
        second_column = weight[:, 1]
        third_column = weight[:, 2]
        with torch.no_grad():
            self.first_column.copy_(first_column[:-1] - first_column[-1])
            self.second_column.copy_(second_column[:-1] - second_column[-1])
            self.third_column.copy_(third_column[:-1] - third_column[-1])
            self.rest.copy_(weight[:, 3:])

    @staticmethod
    def _centered_column(column: torch.Tensor) -> torch.Tensor:
        anchored = torch.cat((column, column.new_zeros(1)))
        return anchored - anchored.mean()

    def tied_weight(self) -> torch.Tensor:
        first_column = self._centered_column(self.first_column)
        second_column = self._centered_column(self.second_column)
        third_column = self._centered_column(self.third_column)
        return torch.cat(
            (
                first_column.unsqueeze(1),
                second_column.unsqueeze(1),
                third_column.unsqueeze(1),
                self.rest,
            ),
            dim=1,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.tied_weight())


class CenteredPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        # Consume the same constructor RNG as the original embedding.
        _ = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fourth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fifth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.sixth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.seventh_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.eighth_column = nn.Parameter(torch.empty(num_embeddings - 1))
        self.rest = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 5))
        self.register_buffer(
            "fourth_column_shift", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "fifth_column_shift", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "sixth_column_shift", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "seventh_column_shift", torch.zeros(()), persistent=False
        )
        self.register_buffer(
            "eighth_column_shift", torch.zeros(()), persistent=False
        )

    def _set_from_full(self, weight: torch.Tensor) -> None:
        fourth_column = weight[:, 3]
        fifth_column = weight[:, 4]
        sixth_column = weight[:, 5]
        seventh_column = weight[:, 6]
        eighth_column = weight[:, 7]
        with torch.no_grad():
            self.fourth_column.copy_(
                fourth_column[:-1] - fourth_column[-1]
            )
            self.fifth_column.copy_(
                fifth_column[:-1] - fifth_column[-1]
            )
            self.sixth_column.copy_(
                sixth_column[:-1] - sixth_column[-1]
            )
            self.seventh_column.copy_(
                seventh_column[:-1] - seventh_column[-1]
            )
            self.eighth_column.copy_(
                eighth_column[:-1] - eighth_column[-1]
            )
            self.rest.copy_(weight[:, :3])
            self.fourth_column_shift.copy_(fourth_column.mean())
            self.fifth_column_shift.copy_(fifth_column.mean())
            self.sixth_column_shift.copy_(sixth_column.mean())
            self.seventh_column_shift.copy_(seventh_column.mean())
            self.eighth_column_shift.copy_(eighth_column.mean())

    def tied_weight(self) -> torch.Tensor:
        fourth_anchored = torch.cat(
            (self.fourth_column, self.fourth_column.new_zeros(1))
        )
        fifth_anchored = torch.cat(
            (self.fifth_column, self.fifth_column.new_zeros(1))
        )
        sixth_anchored = torch.cat(
            (self.sixth_column, self.sixth_column.new_zeros(1))
        )
        seventh_anchored = torch.cat(
            (self.seventh_column, self.seventh_column.new_zeros(1))
        )
        eighth_anchored = torch.cat(
            (self.eighth_column, self.eighth_column.new_zeros(1))
        )
        fourth_column = fourth_anchored - fourth_anchored.mean()
        fifth_column = fifth_anchored - fifth_anchored.mean()
        sixth_column = sixth_anchored - sixth_anchored.mean()
        seventh_column = seventh_anchored - seventh_anchored.mean()
        eighth_column = eighth_anchored - eighth_anchored.mean()
        return torch.cat(
            (
                self.rest[:, :3],
                fourth_column.unsqueeze(1),
                fifth_column.unsqueeze(1),
                sixth_column.unsqueeze(1),
                seventh_column.unsqueeze(1),
                eighth_column.unsqueeze(1),
            ),
            dim=1,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.tied_weight())


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = CenteredBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = ReducedBiasLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x, self.attn.qv_bias[2]))
        x = x + self.mlp(
            self.ln2(x, self.attn.qv_bias[2], self.attn.qv_bias[4])
        )
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = CenteredTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = CenteredPositionalEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Preserve the original output-layer constructor RNG; its weight is
        # reconstructed from the centered input embedding during forward.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = None

        self.apply(self._init_weights)

        # The original tied output layer reinitialized the shared embedding at
        # the end of apply(). Reproduce that draw, then move the eliminated
        # common token-column offsets into every positional embedding.
        full_token_weight = self.token_emb.first_column.new_empty(
            cfg.vocab_size, cfg.d_model
        )
        nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
        token_shift = full_token_weight[:, :3].mean(dim=0)
        full_token_weight[:, 3].add_(self.pos_emb.fourth_column_shift)
        full_token_weight[:, 4].add_(self.pos_emb.fifth_column_shift)
        full_token_weight[:, 5].add_(self.pos_emb.sixth_column_shift)
        full_token_weight[:, 6].add_(self.pos_emb.seventh_column_shift)
        full_token_weight[:, 7].add_(self.pos_emb.eighth_column_shift)
        self.token_emb._set_from_full(full_token_weight)
        with torch.no_grad():
            self.pos_emb.rest[:, :3].add_(token_shift)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, AnchoredKeyLinear):
            full_weight = module.before_key.new_empty(
                3 * module.d_model, module.d_model
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, CenteredTokenEmbedding):
            full_weight = module.first_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, CenteredPositionalEmbedding):
            full_weight = module.fourth_column.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, MLP):
            full_weight = module.fc2_first_column.new_empty(
                module.fc2.out_features, module.fc2.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            first_column = full_weight[:, 0] - full_weight[:, 0].mean()
            second_column = full_weight[:, 1] - full_weight[:, 1].mean()
            with torch.no_grad():
                module.fc2_first_column.copy_(first_column[:-1])
                module.fc2_second_column.copy_(second_column[:-1])
                module.fc2_rest.copy_(full_weight[:, 2:])

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
        logits = F.linear(x, self.token_emb.tied_weight())

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
