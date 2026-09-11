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


class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-16].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, self.weight, bias)


class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias and first four weight columns."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :4]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 4:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 4 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 4
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 4
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, self.bias_basis @ self.bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = KeyAnchoredLinear(d_model)
        self.proj = AttentionGaugeLinear(d_model)
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


class OutputAnchoredLinear(nn.Linear):
    """Linear projection with anchored bias and two zero-mean weight columns."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        basis = self.weight.detach().new_zeros(out_features, out_features - 1)
        for col in range(out_features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("weight_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :2]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 2:].flatten()))
        )
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gauge_size = 2 * (self.out_features - 1)
        leading_weight = self.weight_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 2
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 2
        )
        weight = torch.cat((leading_weight, remaining_weight), dim=1)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))


class NormalizedInputLinear(nn.Linear):
    """Linear map with the final input weight fixed under a zero-mean gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            full_weight[:, :-1] - full_weight[:, -1:]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.pad(self.weight, (0, 1)), self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = NormalizedInputLinear(d_model, d_ff)
        self.fc2 = OutputAnchoredLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class AnchoredLayerNorm(nn.Module):
    """LayerNorm with anchored bias and dynamically tied scale coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)


class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final four scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = AnchoredLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = BiasAnchoredLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PositionAnchoredEmbedding(nn.Embedding):
    """Positional embedding with orthogonal, fixed, and dynamically tied gauges."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

        basis = self.weight.detach().new_zeros(embedding_dim, embedding_dim - 1)
        for col in range(embedding_dim - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("gauge_basis", basis, persistent=False)

        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
        self.leading_gauge_flat_start = (num_embeddings - 9) * embedding_dim
        self.zeroth_gauge_flat_start = (num_embeddings - 8) * embedding_dim
        self.first_gauge_flat_start = (num_embeddings - 7) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 6) * embedding_dim
        self.third_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.fourth_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.earlier_gauge_index = self.earlier_gauge_flat_start
        self.preceding_gauge_index = self.earlier_gauge_index + embedding_dim - 1
        self.leading_gauge_index = self.preceding_gauge_index + embedding_dim - 1
        self.zeroth_gauge_index = self.leading_gauge_index + embedding_dim - 1
        self.first_gauge_index = self.zeroth_gauge_index + embedding_dim - 1
        self.second_gauge_index = self.first_gauge_index + embedding_dim - 1
        self.third_gauge_index = self.second_gauge_index + embedding_dim - 1
        self.fourth_gauge_index = self.third_gauge_index + embedding_dim - 1
        self.gauge_end_index = self.fourth_gauge_index + embedding_dim - 1
        self.tie_index = self.gauge_end_index + embedding_dim - 1
        self.anchor_index = self.tie_index + embedding_dim - 1

        earlier_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.earlier_gauge_flat_start :
                self.earlier_gauge_flat_start + embedding_dim
            ]
        )
        preceding_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.preceding_gauge_flat_start :
                self.preceding_gauge_flat_start + embedding_dim
            ]
        )
        leading_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.leading_gauge_flat_start :
                self.leading_gauge_flat_start + embedding_dim
            ]
        )
        zeroth_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.zeroth_gauge_flat_start :
                self.zeroth_gauge_flat_start + embedding_dim
            ]
        )
        first_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.first_gauge_flat_start :
                self.first_gauge_flat_start + embedding_dim
            ]
        )
        second_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.second_gauge_flat_start :
                self.second_gauge_flat_start + embedding_dim
            ]
        )
        third_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.third_gauge_flat_start :
                self.third_gauge_flat_start + embedding_dim
            ]
        )
        fourth_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.fourth_gauge_flat_start :
                self.fourth_gauge_flat_start + embedding_dim
            ]
        )
        compact_weight = torch.cat(
            (
                flat_weight[: self.earlier_gauge_flat_start],
                earlier_gauge_coords,
                preceding_gauge_coords,
                leading_gauge_coords,
                zeroth_gauge_coords,
                first_gauge_coords,
                second_gauge_coords,
                third_gauge_coords,
                fourth_gauge_coords,
                flat_weight[
                    self.fourth_gauge_flat_start + embedding_dim :
                    self.tie_flat_index
                ],
                flat_weight[
                    self.tie_flat_index + 1 : self.anchor_flat_index
                ],
                flat_weight[self.anchor_flat_index + 1 : -1],
            )
        )
        self.weight = nn.Parameter(compact_weight.clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        earlier_gauge_row = (
            self.gauge_basis
            @ self.weight[self.earlier_gauge_index : self.preceding_gauge_index]
        )
        preceding_gauge_row = (
            self.gauge_basis
            @ self.weight[self.preceding_gauge_index : self.leading_gauge_index]
        )
        leading_gauge_row = (
            self.gauge_basis
            @ self.weight[self.leading_gauge_index : self.zeroth_gauge_index]
        )
        zeroth_gauge_row = (
            self.gauge_basis
            @ self.weight[self.zeroth_gauge_index : self.first_gauge_index]
        )
        first_gauge_row = (
            self.gauge_basis
            @ self.weight[self.first_gauge_index : self.second_gauge_index]
        )
        second_gauge_row = (
            self.gauge_basis
            @ self.weight[self.second_gauge_index : self.third_gauge_index]
        )
        third_gauge_row = (
            self.gauge_basis
            @ self.weight[self.third_gauge_index : self.fourth_gauge_index]
        )
        fourth_gauge_row = (
            self.gauge_basis
            @ self.weight[self.fourth_gauge_index : self.gauge_end_index]
        )
        flat_weight = torch.cat(
            (
                self.weight[: self.earlier_gauge_index],
                earlier_gauge_row,
                preceding_gauge_row,
                leading_gauge_row,
                zeroth_gauge_row,
                first_gauge_row,
                second_gauge_row,
                third_gauge_row,
                fourth_gauge_row,
                self.weight[self.gauge_end_index : self.tie_index],
                self.weight[1:2],
                self.weight[self.tie_index : self.anchor_index],
                self.weight[:1],
                self.weight[self.anchor_index :],
                self.weight.new_zeros(1),
            )
        )
        weight = flat_weight.view(self.num_embeddings, self.embedding_dim)
        return F.embedding(idx, weight)


class TokenAnchoredEmbedding(nn.Embedding):
    """Tied token embedding with one global common-mode scalar anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())

    def full_weight(self) -> torch.Tensor:
        return F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class TiedTokenLinear(nn.Module):
    """Parameterless output projection using the reconstructed token matrix."""

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight)


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = TokenAnchoredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = PositionAnchoredEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with the reconstructed input embedding matrix.
        self.lm_head = TiedTokenLinear()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, NormalizedInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    full_weight[:, :-1] - full_weight[:, -1:]
                )
                if module.bias is not None:
                    module.bias.zero_()
        elif isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            flat_weight = full_weight.flatten()
            earlier_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.earlier_gauge_flat_start :
                    module.earlier_gauge_flat_start + module.embedding_dim
                ]
            )
            preceding_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.preceding_gauge_flat_start :
                    module.preceding_gauge_flat_start + module.embedding_dim
                ]
            )
            leading_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.leading_gauge_flat_start :
                    module.leading_gauge_flat_start + module.embedding_dim
                ]
            )
            zeroth_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.zeroth_gauge_flat_start :
                    module.zeroth_gauge_flat_start + module.embedding_dim
                ]
            )
            first_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.first_gauge_flat_start :
                    module.first_gauge_flat_start + module.embedding_dim
                ]
            )
            second_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.second_gauge_flat_start :
                    module.second_gauge_flat_start + module.embedding_dim
                ]
            )
            third_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.third_gauge_flat_start :
                    module.third_gauge_flat_start + module.embedding_dim
                ]
            )
            fourth_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.fourth_gauge_flat_start :
                    module.fourth_gauge_flat_start + module.embedding_dim
                ]
            )
            compact_weight = torch.cat(
                (
                    flat_weight[: module.earlier_gauge_flat_start],
                    earlier_gauge_coords,
                    preceding_gauge_coords,
                    leading_gauge_coords,
                    zeroth_gauge_coords,
                    first_gauge_coords,
                    second_gauge_coords,
                    third_gauge_coords,
                    fourth_gauge_coords,
                    flat_weight[
                        module.fourth_gauge_flat_start + module.embedding_dim :
                        module.tie_flat_index
                    ],
                    flat_weight[
                        module.tie_flat_index + 1 : module.anchor_flat_index
                    ],
                    flat_weight[module.anchor_flat_index + 1 : -1],
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
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
        logits = self.lm_head(x, self.token_emb.full_weight())

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
