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


class MeanZeroInputLinear(nn.Module):
    """Linear map using a zero-copy chart of the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        self.linear = nn.Linear(in_features - 1, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[..., :-1])


class SingleRotationGaugeQKV(nn.Module):
    """Tied query/key and value map with one centered rotation fixed per head."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        head_dim = d_model // n_head
        if head_dim != 4:
            raise ValueError("single-rotation chart requires four-dimensional heads")

        common = torch.full((head_dim,), 1.0 / math.sqrt(head_dim))
        center0 = torch.tensor((1.0, -1.0, 0.0, 0.0)) / math.sqrt(2.0)
        center1 = torch.tensor((1.0, 1.0, -2.0, 0.0)) / math.sqrt(6.0)
        center2 = torch.tensor((1.0, 1.0, 1.0, -3.0)) / math.sqrt(12.0)
        self.register_buffer(
            "head_basis",
            torch.stack((common, center0, center1, center2), dim=1),
            persistent=False,
        )

        self.qk_common = nn.Parameter(torch.empty(1, in_features))
        self.qk_center0 = nn.Parameter(torch.empty(1, in_features))
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
        self.qk_second_center0 = nn.Parameter(torch.empty(1, in_features))
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            dense_head = torch.empty_like(
                self.qk_common.expand(self.head_basis.size(0), -1)
            )
            nn.init.normal_(dense_head, mean=0.0, std=0.02)
            coeff = self.head_basis.transpose(0, 1) @ dense_head

            first = coeff[1].clone()
            second = coeff[2].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
            coeff[1] = cosine * first + sine * second
            coeff[2] = -sine * first + cosine * second

            first = coeff[1].clone()
            second = coeff[3].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
            coeff[1] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2_tail.copy_(coeff[3:4, 1:])

            dense_second = torch.empty_like(
                self.qk_second_common.expand(self.head_basis.size(0), -1)
            )
            nn.init.normal_(dense_second, mean=0.0, std=0.02)
            second_coeff = self.head_basis.transpose(0, 1) @ dense_second

            first = second_coeff[1].clone()
            second = second_coeff[2].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
            second_coeff[1] = cosine * first + sine * second
            second_coeff[2] = -sine * first + cosine * second

            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_coeff = torch.cat(
            (
                self.qk_common,
                self.qk_center0,
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (1, 0)),
            ),
            dim=0,
        )
        first_head = self.head_basis @ head_coeff
        second_head_coeff = torch.cat(
            (
                self.qk_second_common,
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                self.qk_second_center2,
            ),
            dim=0,
        )
        second_head = self.head_basis @ second_head_coeff
        qk_weight = torch.cat((first_head, second_head), dim=0)
        return F.linear(
            x[..., :-1],
            torch.cat((qk_weight, self.v_weight), dim=0),
        )


class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with scales and bias stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(shared_bias.reshape(1), (0, 7)),
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        positions = torch.arange(max_seq_len)
        relative_distance = (
            positions[:, None] - positions[None, :]
        ).clamp_min(0)
        self.register_buffer(
            "relative_distance", relative_distance, persistent=False
        )
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        qk, v = qkv.chunk(2, dim=-1)

        q = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = qk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        q = q + self.proj.bias.mean()

        relative_distance = self.relative_distance[:seqlen, :seqlen]
        relative_bias = F.pad(self.rel_bias, (1, 0))[:, relative_distance]
        att = (
            (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            + relative_bias
        )
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
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shared_biases: torch.Tensor) -> torch.Tensor:
        output_bias = torch.cat((self.output_bias, shared_biases.reshape(-1)))
        hidden = self.fc1(x) + output_bias.mean()
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        output = F.pad(output, (0, 1))
        return self.drop(output)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = torch.stack(
            (self.attn.proj.bias.mean(), self.attn.proj.bias[0])
        )
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), shared_biases)
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Parameter(
            torch.empty(cfg.vocab_size * cfg.d_model - 1)
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
            normalized_shared_column = self.blocks[-1].attn.proj.weight[:, 1]
            target_mean = 1.0 / math.sqrt(cfg.d_model)
            normalized_shared_column.add_(
                target_mean - normalized_shared_column.mean()
            )
            normalized_shared_bias_column = self.blocks[-1].attn.proj.weight[:, 2]
            normalized_shared_bias_column.sub_(
                normalized_shared_bias_column.mean()
            )
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SingleRotationGaugeQKV):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def token_weight(self) -> torch.Tensor:
        return F.pad(self.token_emb, (1, 0)).view(
            self.cfg.vocab_size, self.cfg.d_model
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, seqlen = idx.shape

        if seqlen > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.cfg.max_seq_len :]
            seqlen = idx.shape[1]

        x = self.drop(F.embedding(idx, self.token_weight()))

        for blk in self.blocks:
            x = blk(x)

        final_proj_weight = self.blocks[-1].attn.proj.weight
        shared_final_scales = torch.stack(
            (
                final_proj_weight[:, 0].mean(),
                math.sqrt(self.cfg.d_model) * final_proj_weight[:, 1].mean(),
            )
        )
        shared_final_bias = (
            math.sqrt(self.cfg.d_model) * final_proj_weight[:, 2].mean()
        )
        x = self.ln_f(x, shared_final_scales, shared_final_bias)
        logits = F.linear(x, self.token_weight())

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
