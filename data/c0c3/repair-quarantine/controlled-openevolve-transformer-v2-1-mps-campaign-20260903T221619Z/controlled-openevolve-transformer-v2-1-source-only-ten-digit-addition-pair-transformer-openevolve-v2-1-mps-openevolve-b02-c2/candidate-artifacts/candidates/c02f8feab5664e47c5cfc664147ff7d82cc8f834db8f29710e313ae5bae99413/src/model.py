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
    """Position embedding with seven translations and five row-shift gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.fixed_coordinates = 7



        position_basis = embedding.weight.new_zeros(
            self.embedding_dim,
            self.embedding_dim - 1,
        )
        for column in range(self.embedding_dim - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            position_basis[: column + 1, column] = 1.0 / denom
            position_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("position_basis", position_basis, persistent=False)

        centered_positions = embedding.weight[1:5] - embedding.weight[1:5].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[5:].detach().clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.position_weight.new_zeros(
            self.embedding_dim - self.fixed_coordinates
        )
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
        compact_positions = (
            self.position_weight @ self.position_basis.transpose(0, 1)
        )
        full_weight = torch.cat(
            (
                origin.unsqueeze(0),
                compact_positions,
                self.weight,
            ),
            dim=0,
        )
        return F.embedding(idx, full_weight)


class CompactQKV(nn.Module):
    """Compact QKV with seven key, four query, and two value-row gauges fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        key_start: int,
        second_head_start: int,
        ln_weight: nn.Parameter,
        fixed_ln_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.second_key_row = second_head_start
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates

        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[
                    second_head_start + 2 : second_head_start + 3
                ],
                linear.weight[second_head_start + 4 : -2],
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

        full_ln_weight = self._full_ln_weight()
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    key_start + 2,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
                    second_head_start + 3,
                ]
            ]
            * full_ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        scaled_query_weight = (
            linear.weight[
                [
                    0,
                    self.head_dim,
                    self.head_dim + 1,
                    self.head_dim + 2,
                ]
            ]
            * full_ln_weight
        )
        centered_query_weight = (
            scaled_query_weight
            - scaled_query_weight.mean(dim=1, keepdim=True)
        )
        self.query_weight = nn.Parameter(
            (centered_query_weight @ basis).detach().clone()
        )

        scaled_value_weight = linear.weight[-2:] * full_ln_weight
        centered_value_weight = (
            scaled_value_weight
            - scaled_value_weight.mean(dim=1, keepdim=True)
        )
        self.value_weight = nn.Parameter(
            (centered_value_weight @ basis).detach().clone()
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

    def _full_ln_weight(self) -> torch.Tensor:
        if self.fixed_ln_weight_coordinates:
            leading_fixed = self.fixed_ln_weight_coordinates - 1
            return torch.cat(
                (
                    self.ln_weight.new_ones(leading_fixed),
                    self.ln_weight,
                    self.ln_weight.new_ones(1),
                )
            )
        return self.ln_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_ln_weight = self._full_ln_weight()
        scaled_query_weight = (
            self.query_weight @ self.key_basis.transpose(0, 1)
        )
        query_weight = scaled_query_weight / full_ln_weight
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / full_ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / full_ln_weight
        second_key_retained_start = self.key_start - 4
        second_key_retained_end = second_key_retained_start + 1
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[
                    self.head_dim - 1 : second_key_retained_start
                ],
                key_weight[: self.head_dim],
                key_weight[self.head_dim : self.head_dim + 2],
                self.weight[
                    second_key_retained_start : second_key_retained_end
                ],
                key_weight[self.head_dim + 2 :],
                self.weight[second_key_retained_end:],
                value_weight,
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
    """LayerNorm with downstream-linear-absorbed affine coordinates fixed."""

    def __init__(
        self,
        layer_norm: nn.LayerNorm,
        fixed_coordinates: int,
        fixed_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.fixed_coordinates = fixed_coordinates
        self.fixed_weight_coordinates = fixed_weight_coordinates
        if fixed_weight_coordinates:
            leading_fixed = fixed_weight_coordinates - 1
            self.weight = nn.Parameter(
                layer_norm.weight[leading_fixed:-1].detach().clone()
            )
        else:
            self.weight = layer_norm.weight
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_weight_coordinates:
            leading_fixed = self.fixed_weight_coordinates - 1
            full_weight = torch.cat(
                (
                    self.weight.new_ones(leading_fixed),
                    self.weight,
                    self.weight.new_ones(1),
                )
            )
        else:
            full_weight = self.weight
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.fixed_coordinates))
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            full_bias,
            self.eps,
        )


class CompactFirstLinearRow(nn.Module):
    """Linear layer with five separated LayerNorm input gauges fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        ln_weight: nn.Parameter,
        fixed_ln_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
        self.quarter_row = linear.out_features // 4
        self.middle_row = linear.out_features // 2
        self.three_quarter_row = 3 * linear.out_features // 4
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.quarter_row],
                linear.weight[self.quarter_row + 1 : self.middle_row],
                linear.weight[
                    self.middle_row + 1 : self.three_quarter_row
                ],
                linear.weight[self.three_quarter_row + 1 : -1],
            ),
            dim=0,
        )
        self.weight = nn.Parameter(retained_weight.detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_rows = (
            linear.weight[
                [
                    0,
                    self.quarter_row,
                    self.middle_row,
                    self.three_quarter_row,
                    -1,
                ]
            ]
            * self._full_ln_weight()
        )
        centered_rows = scaled_rows - scaled_rows.mean(dim=1, keepdim=True)
        self.row_weight = nn.Parameter(
            (centered_rows @ basis).detach().clone()
        )

    def _full_ln_weight(self) -> torch.Tensor:
        if self.fixed_ln_weight_coordinates:
            leading_fixed = self.fixed_ln_weight_coordinates - 1
            return torch.cat(
                (
                    self.ln_weight.new_ones(leading_fixed),
                    self.ln_weight,
                    self.ln_weight.new_ones(1),
                )
            )
        return self.ln_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_rows = self.row_weight @ self.row_basis.transpose(0, 1)
        selected_rows = scaled_rows / self._full_ln_weight()
        quarter_retained_start = self.quarter_row - 1
        middle_retained_start = self.middle_row - 2
        three_quarter_retained_start = self.three_quarter_row - 3
        full_weight = torch.cat(
            (
                selected_rows[:1],
                self.weight[:quarter_retained_start],
                selected_rows[1:2],
                self.weight[
                    quarter_retained_start:middle_retained_start
                ],
                selected_rows[2:3],
                self.weight[
                    middle_retained_start:three_quarter_retained_start
                ],
                selected_rows[3:4],
                self.weight[three_quarter_retained_start:],
                selected_rows[4:],
            ),
            dim=0,
        )
        return F.linear(x, full_weight, self.bias)


class CompactResidualLinear(nn.Module):
    """Linear layer with three weight-column and bias uniform directions fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_columns = (
            linear.weight[:, :3]
            - linear.weight[:, :3].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 3:].detach().clone())

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact_columns = self.bias_basis @ self.column_weight
        full_weight = torch.cat((compact_columns, self.weight), dim=1)
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, full_weight, full_bias)


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
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
            fixed_weight_coordinates=2,
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=2,
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

        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 4.
        with torch.no_grad():
            for coordinate in range(7):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)

            origin_shift = -self.pos_emb.weight[0, 7:].mean()
            self.token_emb.weight[:, :7].add_(origin_shift)
            self.pos_emb.weight[:, :7].sub_(origin_shift)
            self.pos_emb.weight[0].add_(origin_shift)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)

        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient five separated independently biased MLP input rows.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
                block.ln1.fixed_weight_coordinates,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
                block.ln2.fixed_weight_coordinates,
            )
            block.mlp.fc2 = CompactResidualLinear(block.mlp.fc2)

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
