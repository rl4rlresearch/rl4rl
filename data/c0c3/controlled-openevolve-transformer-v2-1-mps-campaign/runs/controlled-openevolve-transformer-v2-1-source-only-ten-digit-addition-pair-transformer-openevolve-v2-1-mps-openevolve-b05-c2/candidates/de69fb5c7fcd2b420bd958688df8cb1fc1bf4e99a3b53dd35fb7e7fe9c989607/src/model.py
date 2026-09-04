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


def _mean_zero_basis(dim: int) -> torch.Tensor:
    """Orthonormal basis for vectors whose coordinates sum to zero."""
    if dim < 2:
        raise ValueError("mean-zero features require dimension >= 2")
    basis = torch.zeros(dim, dim - 1)
    for col in range(dim - 1):
        scale = 1.0 / math.sqrt((col + 1) * (col + 2))
        basis[: col + 1, col] = scale
        basis[col + 1, col] = -(col + 1) * scale
    return basis


class MeanZeroEmbedding(nn.Module):
    """Embedding with its LayerNorm-invisible common mode removed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        self.weight = nn.Parameter(full_weight @ basis)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight) @ self.basis.transpose(0, 1)


class GaugeFixedEmbedding(nn.Module):
    """Tied embedding with every token/position common-shift gauge fixed."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        basis = _mean_zero_basis(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(full_weight)
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ basis
        shift = last_coords
        gauged_weight = gauged_weight - basis @ shift
        self.weight_rows = nn.Parameter(gauged_weight[:-1])
        self.register_buffer(
            "initial_position_shift", shift.detach().clone(), persistent=False
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        anchor = full_weight[-1].mean()
        gauged_weight = full_weight - anchor
        last_coords = gauged_weight[-1] @ self.basis
        shift = last_coords
        gauged_weight = gauged_weight - self.basis @ shift
        with torch.no_grad():
            self.weight_rows.copy_(gauged_weight[:-1])
            self.initial_position_shift.copy_(shift)

    @property
    def weight(self) -> torch.Tensor:
        last_row = self.weight_rows.new_zeros(self.embedding_dim)
        return torch.cat((self.weight_rows, last_row.unsqueeze(0)), dim=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight)


class TiedOutputLinear(nn.Module):
    """Parameter-free output view of a gauge-fixed embedding."""
    def __init__(self, embedding: GaugeFixedEmbedding):
        super().__init__()
        object.__setattr__(self, "embedding", embedding)

        discarded_weight = torch.empty(
            embedding.num_embeddings, embedding.embedding_dim
        )
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MeanZeroInputLinear(nn.Linear):
    """Linear map modulo the common input mode removed by LayerNorm."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(in_features)
        self.weight = nn.Parameter(self.weight.detach() @ basis)
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight @ self.basis.transpose(0, 1), self.bias)


class ElevenRotationTenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and ten value-output gauges fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("eleven-rotation gauge fixing requires two suitable heads")
        self.in_features = in_features
        self.out_features = out_features
        self.second_query = head_dim
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        fixed_weight, value_rotations = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_value_rotations",
            value_rotations.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
        self.second_weight = nn.Parameter(fixed_weight[1, 2:])
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
        self.head_two_second_weight = nn.Parameter(
            fixed_weight[self.second_query + 1, 2:]
        )
        self.head_two_third_weight = nn.Parameter(
            fixed_weight[self.second_query + 2, 1:]
        )
        self.pre_value_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:2 * self.in_features]
        )
        value_start = 2 * self.in_features
        self.first_value_weight = nn.Parameter(
            fixed_weight[value_start, 2:]
        )
        self.second_value_weight = nn.Parameter(
            fixed_weight[value_start + 1, 2:]
        )
        self.third_value_weight = nn.Parameter(
            fixed_weight[value_start + 2, 1:]
        )
        self.first_head_value_tail = nn.Parameter(
            fixed_weight[value_start + 3:value_start + self.second_query]
        )
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 2:]
        )
        second_row = value_start + self.second_query + 1
        self.head_two_value_second_weight = nn.Parameter(
            fixed_weight[second_row, 2:]
        )
        third_row = value_start + self.second_query + 2
        self.head_two_value_third_weight = nn.Parameter(
            fixed_weight[third_row, 1:]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)

    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        fixed_weight = full_weight.clone()
        rotations = (
            (0, 0),
            (1, 0),
            (2, 0),
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (0, 1),
            (self.second_query, 1),
            (1, 1),
            (self.second_query + 1, 1),
            (0, 2),
        )
        for query_start, input_coord in rotations:
            pivot = fixed_weight[
                query_start:query_start + 2, input_coord
            ]
            radius = pivot.norm().clamp_min(
                torch.finfo(full_weight.dtype).tiny
            )
            cosine = pivot[1] / radius
            sine = -pivot[0] / radius
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            fixed_weight[query_start:query_start + 2] = (
                rotation @ fixed_weight[query_start:query_start + 2]
            )
            key_start = self.in_features + query_start
            fixed_weight[key_start:key_start + 2] = (
                rotation @ fixed_weight[key_start:key_start + 2]
            )
            fixed_weight[query_start, input_coord] = 0.0

        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (self.second_query, 1),
            (self.second_query + 1, 1),
        ):
            row_start = value_start + value_offset
            pivot = fixed_weight[
                row_start:row_start + 2, input_coord
            ]
            radius = pivot.norm().clamp_min(
                torch.finfo(full_weight.dtype).tiny
            )
            cosine = pivot[1] / radius
            sine = -pivot[0] / radius
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            fixed_weight[row_start:row_start + 2] = (
                rotation @ fixed_weight[row_start:row_start + 2]
            )
            fixed_weight[row_start, input_coord] = 0.0
            value_rotations.append(rotation)

        return fixed_weight, torch.stack(value_rotations)

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight, value_rotations = self._gauge_fix(full_weight)
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 3:])
            self.second_weight.copy_(fixed_weight[1, 2:])
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
            self.head_two_second_weight.copy_(
                fixed_weight[self.second_query + 1, 2:]
            )
            self.head_two_third_weight.copy_(
                fixed_weight[self.second_query + 2, 1:]
            )
            self.pre_value_weight.copy_(
                fixed_weight[self.second_query + 3:2 * self.in_features]
            )
            value_start = 2 * self.in_features
            self.first_value_weight.copy_(
                fixed_weight[value_start, 2:]
            )
            self.second_value_weight.copy_(
                fixed_weight[value_start + 1, 2:]
            )
            self.third_value_weight.copy_(
                fixed_weight[value_start + 2, 1:]
            )
            self.first_head_value_tail.copy_(
                fixed_weight[
                    value_start + 3:value_start + self.second_query
                ]
            )
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 2:]
            )
            second_row = value_start + self.second_query + 1
            self.head_two_value_second_weight.copy_(
                fixed_weight[second_row, 2:]
            )
            third_row = value_start + self.second_query + 2
            self.head_two_value_third_weight.copy_(
                fixed_weight[third_row, 1:]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_value_rotations.copy_(value_rotations)

    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (3, 0))
        second_row = F.pad(self.second_weight, (2, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (2, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (2, 0))
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        first_value_row = F.pad(self.first_value_weight, (2, 0))
        second_value_row = F.pad(self.second_value_weight, (2, 0))
        third_value_row = F.pad(self.third_value_weight, (1, 0))
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
        head_two_value_second_row = F.pad(
            self.head_two_value_second_weight, (2, 0)
        )
        head_two_value_third_row = F.pad(
            self.head_two_value_third_weight, (1, 0)
        )
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
                head_two_second_row.unsqueeze(0),
                head_two_third_row.unsqueeze(0),
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                third_value_row.unsqueeze(0),
                self.first_head_value_tail,
                head_two_value_row.unsqueeze(0),
                head_two_value_second_row.unsqueeze(0),
                head_two_value_third_row.unsqueeze(0),
                last_row.unsqueeze(0),
            ),
            dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(out_features)
        self.weight = nn.Parameter(basis.transpose(0, 1) @ self.weight.detach())
        self.bias = nn.Parameter(basis.transpose(0, 1) @ self.bias.detach())
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.basis @ self.weight, self.basis @ self.bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = ElevenRotationTenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def apply_initial_value_rotations(self) -> None:
        with torch.no_grad():
            column_starts = (
                0,
                1,
                2,
                0,
                1,
                self.head_dim,
                self.head_dim + 1,
                self.head_dim + 2,
                self.head_dim,
                self.head_dim + 1,
            )
            for rotation, column_start in zip(
                self.qkv.initial_value_rotations,
                column_starts,
            ):
                columns = (
                    self.proj.weight[:, column_start:column_start + 2]
                    @ rotation.transpose(0, 1)
                )
                self.proj.weight[
                    :, column_start:column_start + 2
                ].copy_(columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 3:],
            )
        )
        qkv_bias = torch.cat(
            (
                q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
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
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class SeptupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with seven scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 7), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
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
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = TiedOutputLinear(self.token_emb)

        self.apply(self._init_weights)

        for block in self.blocks:
            block.attn.apply_initial_value_rotations()

        with torch.no_grad():
            self.pos_emb.weight.add_(
                self.token_emb.initial_position_shift
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            full_weight = module.weight_rows.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, TiedOutputLinear):
            embedding = module.embedding
            full_weight = embedding.weight_rows.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            embedding.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.basis)
        elif isinstance(module, ElevenRotationTenValueGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
        elif isinstance(module, MeanZeroInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.basis)
                module.bias.zero_()
        elif isinstance(module, MeanZeroOutputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    module.basis.transpose(0, 1) @ full_weight
                )
                module.bias.zero_()
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
