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
        if self.head_dim < 2:
            raise ValueError("value-basis compaction requires head dimension at least two")
        self.d_model = d_model

        # Align each head's first value-weight column with one basis vector.
        # The resulting three-dimensional zero subspace in the first head can
        # then be triangularized through two second-column rotations.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1, 0))
            value_fixed_coordinates.append((head, 0, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3, 0))
                value_fixed_coordinates.append((head, 2, 0))
                value_basis_rotations.append((head, 1, 3, 0))
                value_fixed_coordinates.append((head, 1, 0))
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                + input_column
                for head, local, input_column in value_fixed_coordinates
            )
        )

        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from attention softmax. A shared value
        # bias passes unchanged through attention and is absorbable by the
        # retained output-projection bias, so store only the query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)

        # A uniform projection-bias shift survives the residual connection but
        # is erased by both subsequent LayerNorms. Store only its zero-sum part.
        self.proj.bias = nn.Parameter(self.proj.bias.new_zeros(d_model - 1))
        proj_bias_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            proj_bias_basis[: col + 1, col] = 1.0 / scale
            proj_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("proj_bias_basis", proj_bias_basis, persistent=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compact_value_basis(self) -> None:
        # Rotate value coordinates and counter-rotate their output columns.
        # Also quotient every projection column except optimization-sensitive
        # column three.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for (
                head,
                first_local,
                second_local,
                input_column,
            ) in self.value_basis_rotations:
                first_value = (
                    2 * self.d_model + head * self.head_dim + first_local
                )
                second_value = (
                    2 * self.d_model + head * self.head_dim + second_local
                )
                first_column = head * self.head_dim + first_local
                second_column = head * self.head_dim + second_local

                a = qkv_weight[first_value, input_column]
                b = qkv_weight[second_value, input_column]
                norm = torch.hypot(a, b)
                cosine = b / norm
                sine = a / norm

                row0 = qkv_weight[first_value].clone()
                row1 = qkv_weight[second_value].clone()
                qkv_weight[first_value] = cosine * row0 - sine * row1
                qkv_weight[second_value] = sine * row0 + cosine * row1

                col0 = proj_weight[:, first_column].clone()
                col1 = proj_weight[:, second_column].clone()
                proj_weight[:, first_column] = cosine * col0 - sine * col1
                proj_weight[:, second_column] = sine * col0 + cosine * col1

            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
                pieces.append(flat_weight[start:fixed_index])
                start = fixed_index + 1
            pieces.append(flat_weight[start:])
            compact_weight = torch.cat(pieces)

            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1)
                @ proj_weight[:, (0, 1, 2, 4, 5, 6, 7)]
            )
            remaining_proj_weight = proj_weight[:, 3:4].clone()

        self.qkv.weight = nn.Parameter(compact_weight)
        self.proj_compact_columns = nn.Parameter(compact_proj_columns)
        self.proj.weight = nn.Parameter(remaining_proj_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
        weight_pieces = []
        compact_start = 0
        for removed, fixed_index in enumerate(self.value_fixed_indices):
            compact_index = fixed_index - removed
            weight_pieces.append(self.qkv.weight[compact_start:compact_index])
            weight_pieces.append(self.qkv.weight.new_zeros(1))
            compact_start = compact_index
        weight_pieces.append(self.qkv.weight[compact_start:])
        qkv_weight = torch.cat(weight_pieces).view(
            3 * self.d_model, self.d_model
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
        proj_bias = self.proj_bias_basis @ self.proj.bias
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
        proj_weight = torch.cat(
            (
                compact_columns[:, :3],
                self.proj.weight,
                compact_columns[:, 3:],
            ),
            dim=1,
        )
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        # The following final LayerNorm removes a uniform residual-channel
        # shift, so parameterize this bias in an orthonormal zero-sum basis.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def compact_output_columns(self) -> None:
        # Uniform output components are erased by the final LayerNorm.
        # Quotient every column except column three, which remains an
        # unrestricted optimization coordinate.
        with torch.no_grad():
            weight = self.fc2.weight.detach()
            compact_columns = (
                self.bias_basis.transpose(0, 1)
                @ weight[:, (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11)]
            )
            remaining_weight = weight[:, 3:4].clone()
        self.fc2_compact_columns = nn.Parameter(compact_columns)
        self.fc2.weight = nn.Parameter(remaining_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        compact_columns = self.bias_basis @ self.fc2_compact_columns
        weight = torch.cat(
            (
                compact_columns[:, :3],
                self.fc2.weight,
                compact_columns[:, 3:],
            ),
            dim=1,
        )
        return self.drop(F.linear(hidden, weight, bias))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # A uniform pre-attention shift is absorbable by the retained query
        # and output-projection biases; key shifts cancel from the softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias. Remove only that single direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ln1_bias = self.ln1_bias_basis @ self.ln1.bias
        normalized = F.layer_norm(
            x,
            self.ln1.normalized_shape,
            self.ln1.weight,
            ln1_bias,
            self.ln1.eps,
        )
        x = x + self.attn(normalized)

        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            self.ln2.weight,
            ln2_bias,
            self.ln2.eps,
        )
        x = x + self.mlp(normalized)
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
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()

        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # row 14, and the final nine rows. Keep sensitive rows 7 and 13
        # unrestricted.
        self.compact_pos_count = 22
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:13],
                    initialized_pos[14:15],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (
                    initialized_pos[7:8],
                    initialized_pos[13:14],
                    initialized_pos[15:-9],
                ),
                dim=0,
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)

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
        compact_size = self.compact_pos_count * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_count, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        unrestricted_rows = self.pos_emb.weight[compact_size:].view(
            self.cfg.max_seq_len - self.compact_pos_count,
            self.cfg.d_model,
        )
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:12],
                unrestricted_rows[1:2],
                compact_rows[12:13],
                unrestricted_rows[2:],
                compact_rows[13:],
            ),
            dim=0,
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
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
