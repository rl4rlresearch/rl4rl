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


class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with all position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_first_common", torch.empty(()), persistent=False
        )
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fifth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_third_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fourth_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fifth_common", torch.empty(()), persistent=False
        )

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
        for j in range(embedding_dim - 2):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        position_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            position_basis[: j + 1, j] = 1.0 / scale
            position_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("position_basis", position_basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first_weight = (self.position_basis @ self.weight).unsqueeze(1)
        remaining = F.embedding(idx, first_weight)
        second_weight = (
            self.position_basis @ self.second_coordinate
        ).unsqueeze(1)
        second = F.embedding(idx, second_weight)
        third_weight = (
            self.position_basis @ self.third_coordinate
        ).unsqueeze(1)
        third = F.embedding(idx, third_weight)
        fourth_weight = (
            self.position_basis @ self.fourth_coordinate
        ).unsqueeze(1)
        fourth = F.embedding(idx, fourth_weight)
        fifth_weight = (
            self.position_basis @ self.fifth_coordinate
        ).unsqueeze(1)
        fifth = F.embedding(idx, fifth_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (
                remaining,
                second,
                third,
                fourth,
                fifth,
                last,
            ),
            dim=-1,
        )
        return coordinates @ self.basis.transpose(0, 1)


class MeanFreeTokenEmbedding(nn.Embedding):
    """Globally mean-free tied embedding with isolated token-row means."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()

        content_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            content_basis[: j + 1, j] = 1.0 / scale
            content_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("content_basis", content_basis, persistent=False)

        mean_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            mean_basis[: j + 1, j] = 1.0 / scale
            mean_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("mean_basis", mean_basis, persistent=False)

        row_average = full_weight.mean(dim=1)
        self.weight = nn.Parameter(
            ((full_weight - row_average.unsqueeze(1)) @ content_basis).clone()
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def full_weight(self) -> torch.Tensor:
        centered = self.weight @ self.content_basis.transpose(0, 1)
        row_offsets = (
            (self.mean_basis @ self.row_mean).unsqueeze(1)
            / math.sqrt(self.embedding_dim)
        )
        return centered + row_offsets

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class TiedMeanFreeOutput(nn.Linear):
    """Output projection sharing the quotient embedding's parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        embedding: MeanFreeTokenEmbedding,
    ):
        # Preserve the constructor draw made by the original output Linear.
        super().__init__(in_features, out_features, bias=False)
        del self.weight
        object.__setattr__(self, "embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
        self.proj = MeanFreeResidualLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
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


class MeanFreeResidualLinear(nn.Linear):
    """Residual linear map modulo output components removed by LayerNorm."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()

        basis = torch.zeros(out_features, out_features - 1)
        for j in range(out_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        self.weight = nn.Parameter((basis.transpose(0, 1) @ full_weight).clone())
        self.bias = nn.Parameter((full_bias @ basis).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.bias_basis @ self.weight
        full_bias = self.bias @ self.bias_basis.transpose(0, 1)
        return F.linear(x, full_weight, full_bias)


class DistributedTwelvePrunedInputWeightLinear(nn.Linear):
    """Linear map with twelve quotients distributed across input coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_three_rows = nn.Parameter(full_weight[:3, :-1].clone())
        self.fourth_row = nn.Parameter(
            torch.cat((full_weight[3, :3], full_weight[3, 4:])).clone()
        )
        self.weight = nn.Parameter(
            torch.cat((full_weight[5:6, :1], full_weight[5:6, 2:]), dim=1).clone()
        )
        self.fifth_row = nn.Parameter(
            torch.cat((full_weight[4, :2], full_weight[4, 3:])).clone()
        )
        self.seventh_row = nn.Parameter(
            torch.cat((full_weight[6, :1], full_weight[6, 2:])).clone()
        )
        self.eighth_row = nn.Parameter(full_weight[7, 1:].clone())
        self.last_four_rows = nn.Parameter(full_weight[-4:, :-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_three_rows = F.pad(self.first_three_rows, (0, 1))
        fourth_row = torch.cat(
            (
                self.fourth_row[:3],
                self.fourth_row.new_zeros(1),
                self.fourth_row[3:],
            )
        ).unsqueeze(0)
        seventh_row = torch.cat(
            (
                self.seventh_row[:1],
                self.seventh_row.new_zeros(1),
                self.seventh_row[1:],
            )
        ).unsqueeze(0)
        eighth_row = F.pad(self.eighth_row, (1, 0)).unsqueeze(0)
        fifth_row = torch.cat(
            (
                self.fifth_row[:2],
                self.fifth_row.new_zeros(1),
                self.fifth_row[2:],
            )
        ).unsqueeze(0)
        sixth_row = torch.cat(
            (
                self.weight[:, :1],
                self.weight.new_zeros((1, 1)),
                self.weight[:, 1:],
            ),
            dim=1,
        )
        last_four_rows = F.pad(self.last_four_rows, (0, 1))
        weight = torch.cat(
            (
                first_three_rows,
                fourth_row,
                fifth_row,
                sixth_row,
                seventh_row,
                eighth_row,
                last_four_rows,
            ),
            dim=0,
        )
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = DistributedTwelvePrunedInputWeightLinear(d_model, d_ff)
        self.fc2 = MeanFreeResidualLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and five scales fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (full_weight[:1], full_weight[3:4], full_weight[6:-1])
            ).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:2],
                self.weight.new_ones(2),
                self.weight[2:],
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)


class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its bias absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        self.bias = None
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.fixed_bias, self.eps
        )


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = FourPrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = BiasFreeMLPLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = MeanFreeTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = QuotientFinalLayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = TiedMeanFreeOutput(
            cfg.d_model, cfg.vocab_size, self.token_emb
        )

        self.apply(self._init_weights)

        # Preserve the initialized token-plus-position inputs after removing
        # positional common modes. The corresponding token shifts are also
        # softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(
                self.pos_emb.removed_first_common
            )
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_second_common
            )
            self.token_emb.weight[:, 2].add_(
                self.pos_emb.removed_third_common
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb.removed_fourth_common
            )
            self.token_emb.weight[:, 4].add_(
                self.pos_emb.removed_fifth_common
            )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, TiedMeanFreeOutput):
            # The original tied output Linear initialized the shared embedding a
            # second time. Preserve that draw and retain its quotient coordinates.
            embedding = module.embedding
            with torch.no_grad():
                full = torch.empty(
                    embedding.num_embeddings,
                    embedding.embedding_dim,
                    device=embedding.weight.device,
                    dtype=embedding.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full @ embedding.content_basis)
                embedding.row_mean.zero_()
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full @ module.content_basis)
                module.row_mean.zero_()
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from all
            # retained content coordinates.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(projected[:, 0] @ module.position_basis)
                module.removed_first_common.copy_(projected[:, 0].mean())
                module.second_coordinate.copy_(
                    projected[:, 1] @ module.position_basis
                )
                module.third_coordinate.copy_(
                    projected[:, 2] @ module.position_basis
                )
                module.fourth_coordinate.copy_(
                    projected[:, 3] @ module.position_basis
                )
                module.fifth_coordinate.copy_(
                    projected[:, 4] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
                module.removed_third_common.copy_(projected[:, 2].mean())
                module.removed_fourth_common.copy_(projected[:, 3].mean())
                module.removed_fifth_common.copy_(projected[:, 4].mean())
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
            # LayerNorm scale, canonicalize rows zero through two and the last
            # four rows on the final coordinate, row three on the fourth, row
            # four on the third, rows five and six on the second, and row seven
            # on the first coordinate.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_three_rows.copy_(
                    full[:3, :-1] - full[:3, -1].unsqueeze(1)
                )
                module.fourth_row.copy_(
                    torch.cat(
                        (
                            full[3, :3] - full[3, 3],
                            full[3, 4:] - full[3, 3],
                        )
                    )
                )
                module.weight.copy_(
                    torch.cat(
                        (
                            full[5:6, :1] - full[5:6, 1:2],
                            full[5:6, 2:] - full[5:6, 1:2],
                        ),
                        dim=1,
                    )
                )
                module.fifth_row.copy_(
                    torch.cat(
                        (
                            full[4, :2] - full[4, 2],
                            full[4, 3:] - full[4, 2],
                        )
                    )
                )
                module.seventh_row.copy_(
                    torch.cat(
                        (
                            full[6, :1] - full[6, 1],
                            full[6, 2:] - full[6, 1],
                        )
                    )
                )
                module.eighth_row.copy_(full[7, 1:] - full[7, 0])
                module.last_four_rows.copy_(
                    full[-4:, :-1] - full[-4:, -1].unsqueeze(1)
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, MeanFreeResidualLinear):
            # Draw the original full output matrix so subsequent initialization
            # keeps the same RNG sequence, then retain its observable component.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(module.bias_basis.transpose(0, 1) @ full)
                nn.init.zeros_(module.bias)
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
