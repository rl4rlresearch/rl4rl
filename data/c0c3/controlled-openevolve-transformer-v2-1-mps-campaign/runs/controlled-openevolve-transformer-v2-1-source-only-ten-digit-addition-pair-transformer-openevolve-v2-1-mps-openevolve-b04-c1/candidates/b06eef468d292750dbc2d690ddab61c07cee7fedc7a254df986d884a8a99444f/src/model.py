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


class OneRowInputGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.second_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.third_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.fourth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.fifth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.sixth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.seventh_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.eighth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.ninth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.tenth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.eleventh_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.twelfth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.thirteenth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.fourteenth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.fifteenth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.sixteenth_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.rest_weight = nn.Parameter(
            torch.empty(out_features - 16, in_features)
        )

        inv_sqrt = in_features ** -0.5
        reflector = torch.full((in_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Consume the same constructor-time draw as the replaced Linear.
        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        self._set_weight(conceptual_weight)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * torch.dot(self.reflector, value) / self.reflector_norm_sq
        return value - self.reflector * projection

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed_first = self._householder(conceptual_weight[0])
            self.first_coordinates.copy_(transformed_first[1:])
            transformed_second = self._householder(conceptual_weight[1])
            self.second_coordinates.copy_(transformed_second[1:])
            transformed_third = self._householder(conceptual_weight[2])
            self.third_coordinates.copy_(transformed_third[1:])
            transformed_fourth = self._householder(conceptual_weight[3])
            self.fourth_coordinates.copy_(transformed_fourth[1:])
            transformed_fifth = self._householder(conceptual_weight[4])
            self.fifth_coordinates.copy_(transformed_fifth[1:])
            transformed_sixth = self._householder(conceptual_weight[5])
            self.sixth_coordinates.copy_(transformed_sixth[1:])
            transformed_seventh = self._householder(conceptual_weight[6])
            self.seventh_coordinates.copy_(transformed_seventh[1:])
            transformed_eighth = self._householder(conceptual_weight[7])
            self.eighth_coordinates.copy_(transformed_eighth[1:])
            transformed_ninth = self._householder(conceptual_weight[8])
            self.ninth_coordinates.copy_(transformed_ninth[1:])
            transformed_tenth = self._householder(conceptual_weight[9])
            self.tenth_coordinates.copy_(transformed_tenth[1:])
            transformed_eleventh = self._householder(conceptual_weight[10])
            self.eleventh_coordinates.copy_(transformed_eleventh[1:])
            transformed_twelfth = self._householder(conceptual_weight[11])
            self.twelfth_coordinates.copy_(transformed_twelfth[1:])
            transformed_thirteenth = self._householder(conceptual_weight[12])
            self.thirteenth_coordinates.copy_(transformed_thirteenth[1:])
            transformed_fourteenth = self._householder(conceptual_weight[13])
            self.fourteenth_coordinates.copy_(transformed_fourteenth[1:])
            transformed_fifteenth = self._householder(conceptual_weight[14])
            self.fifteenth_coordinates.copy_(transformed_fifteenth[1:])
            transformed_sixteenth = self._householder(conceptual_weight[15])
            self.sixteenth_coordinates.copy_(transformed_sixteenth[1:])
            self.rest_weight.copy_(conceptual_weight[16:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.rest_weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        first = self._householder(first)
        second = F.pad(self.second_coordinates, (1, 0))
        second = self._householder(second)
        third = F.pad(self.third_coordinates, (1, 0))
        third = self._householder(third)
        fourth = F.pad(self.fourth_coordinates, (1, 0))
        fourth = self._householder(fourth)
        fifth = F.pad(self.fifth_coordinates, (1, 0))
        fifth = self._householder(fifth)
        sixth = F.pad(self.sixth_coordinates, (1, 0))
        sixth = self._householder(sixth)
        seventh = F.pad(self.seventh_coordinates, (1, 0))
        seventh = self._householder(seventh)
        eighth = F.pad(self.eighth_coordinates, (1, 0))
        eighth = self._householder(eighth)
        ninth = F.pad(self.ninth_coordinates, (1, 0))
        ninth = self._householder(ninth)
        tenth = F.pad(self.tenth_coordinates, (1, 0))
        tenth = self._householder(tenth)
        eleventh = F.pad(self.eleventh_coordinates, (1, 0))
        eleventh = self._householder(eleventh)
        twelfth = F.pad(self.twelfth_coordinates, (1, 0))
        twelfth = self._householder(twelfth)
        thirteenth = F.pad(self.thirteenth_coordinates, (1, 0))
        thirteenth = self._householder(thirteenth)
        fourteenth = F.pad(self.fourteenth_coordinates, (1, 0))
        fourteenth = self._householder(fourteenth)
        fifteenth = F.pad(self.fifteenth_coordinates, (1, 0))
        fifteenth = self._householder(fifteenth)
        sixteenth = F.pad(self.sixteenth_coordinates, (1, 0))
        sixteenth = self._householder(sixteenth)
        weight = torch.cat(
            (
                first.unsqueeze(0),
                second.unsqueeze(0),
                third.unsqueeze(0),
                fourth.unsqueeze(0),
                fifth.unsqueeze(0),
                sixth.unsqueeze(0),
                seventh.unsqueeze(0),
                eighth.unsqueeze(0),
                ninth.unsqueeze(0),
                tenth.unsqueeze(0),
                eleventh.unsqueeze(0),
                twelfth.unsqueeze(0),
                thirteenth.unsqueeze(0),
                fourteenth.unsqueeze(0),
                fifteenth.unsqueeze(0),
                sixteenth.unsqueeze(0),
                self.rest_weight,
            ),
            dim=0,
        )
        return F.linear(x, weight)


class OneColumnGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))
        self.third_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.fourth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.fifth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.sixth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.seventh_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.eighth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.ninth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.tenth_coordinates = nn.Parameter(torch.empty(out_features - 1))
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
            transformed_third = self._householder(conceptual_rest[:, 0])
            self.third_coordinates.copy_(transformed_third[1:])
            transformed_fourth = self._householder(conceptual_rest[:, 1])
            self.fourth_coordinates.copy_(transformed_fourth[1:])
            transformed_fifth = self._householder(conceptual_rest[:, 2])
            self.fifth_coordinates.copy_(transformed_fifth[1:])
            transformed_sixth = self._householder(conceptual_rest[:, 3])
            self.sixth_coordinates.copy_(transformed_sixth[1:])
            transformed_seventh = self._householder(conceptual_rest[:, 4])
            self.seventh_coordinates.copy_(transformed_seventh[1:])
            transformed_eighth = self._householder(conceptual_rest[:, 5])
            self.eighth_coordinates.copy_(transformed_eighth[1:])
            transformed_ninth = self._householder(conceptual_rest[:, 6])
            self.ninth_coordinates.copy_(transformed_ninth[1:])
            transformed_tenth = self._householder(conceptual_rest[:, 7])
            self.tenth_coordinates.copy_(transformed_tenth[1:])
            transformed_middle = self._householder(conceptual_rest[:, -2])
            self.middle_coordinates.copy_(transformed_middle[1:])
            transformed_last = self._householder(conceptual_rest[:, -1])
            self.last_coordinates.copy_(transformed_last[1:])

    def reset_rest_parameters(self) -> None:
        conceptual_rest = self.tenth_coordinates.new_empty(
            self.out_features, self.in_features - 2
        )
        nn.init.normal_(conceptual_rest, mean=0.0, std=0.02)
        self._set_rest(conceptual_rest)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        first = self._householder(first)
        third = F.pad(self.third_coordinates, (1, 0))
        third = self._householder(third)
        fourth = F.pad(self.fourth_coordinates, (1, 0))
        fourth = self._householder(fourth)
        fifth = F.pad(self.fifth_coordinates, (1, 0))
        fifth = self._householder(fifth)
        sixth = F.pad(self.sixth_coordinates, (1, 0))
        sixth = self._householder(sixth)
        seventh = F.pad(self.seventh_coordinates, (1, 0))
        seventh = self._householder(seventh)
        eighth = F.pad(self.eighth_coordinates, (1, 0))
        eighth = self._householder(eighth)
        ninth = F.pad(self.ninth_coordinates, (1, 0))
        ninth = self._householder(ninth)
        tenth = F.pad(self.tenth_coordinates, (1, 0))
        tenth = self._householder(tenth)
        middle = F.pad(self.middle_coordinates, (1, 0))
        middle = self._householder(middle)
        last = F.pad(self.last_coordinates, (1, 0))
        last = self._householder(last)
        weight = torch.cat(
            (
                first.transpose(0, 1),
                third.unsqueeze(1),
                fourth.unsqueeze(1),
                fifth.unsqueeze(1),
                sixth.unsqueeze(1),
                seventh.unsqueeze(1),
                eighth.unsqueeze(1),
                ninth.unsqueeze(1),
                tenth.unsqueeze(1),
                middle.unsqueeze(1),
                last.unsqueeze(1),
            ),
            dim=1,
        )
        return F.linear(x, weight)


class OneColumnShiftGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.second_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.third_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.fourth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.fifth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.sixth_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.seventh_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.eighth_coordinates = nn.Parameter(torch.empty(out_features - 1))

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Consume the same constructor-time random draw as Linear so all
        # subsequent modules retain the verified initialization stream.
        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        self._set_weight(conceptual_weight)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * torch.dot(self.reflector, value) / self.reflector_norm_sq
        return value - self.reflector * projection

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed_first = self._householder(conceptual_weight[:, 0])
            self.first_coordinates.copy_(transformed_first[1:])
            transformed_second = self._householder(conceptual_weight[:, 1])
            self.second_coordinates.copy_(transformed_second[1:])
            transformed_third = self._householder(conceptual_weight[:, 2])
            self.third_coordinates.copy_(transformed_third[1:])
            transformed_fourth = self._householder(conceptual_weight[:, 3])
            self.fourth_coordinates.copy_(transformed_fourth[1:])
            transformed_fifth = self._householder(conceptual_weight[:, 4])
            self.fifth_coordinates.copy_(transformed_fifth[1:])
            transformed_sixth = self._householder(conceptual_weight[:, 5])
            self.sixth_coordinates.copy_(transformed_sixth[1:])
            transformed_seventh = self._householder(conceptual_weight[:, 6])
            self.seventh_coordinates.copy_(transformed_seventh[1:])
            transformed_eighth = self._householder(conceptual_weight[:, 7])
            self.eighth_coordinates.copy_(transformed_eighth[1:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.eighth_coordinates.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        first = self._householder(first)
        second = F.pad(self.second_coordinates, (1, 0))
        second = self._householder(second)
        third = F.pad(self.third_coordinates, (1, 0))
        third = self._householder(third)
        fourth = F.pad(self.fourth_coordinates, (1, 0))
        fourth = self._householder(fourth)
        fifth = F.pad(self.fifth_coordinates, (1, 0))
        fifth = self._householder(fifth)
        sixth = F.pad(self.sixth_coordinates, (1, 0))
        sixth = self._householder(sixth)
        seventh = F.pad(self.seventh_coordinates, (1, 0))
        seventh = self._householder(seventh)
        eighth = F.pad(self.eighth_coordinates, (1, 0))
        eighth = self._householder(eighth)
        weight = torch.cat(
            (
                first.unsqueeze(1),
                second.unsqueeze(1),
                third.unsqueeze(1),
                fourth.unsqueeze(1),
                fifth.unsqueeze(1),
                sixth.unsqueeze(1),
                seventh.unsqueeze(1),
                eighth.unsqueeze(1),
            ),
            dim=1,
        )
        return F.linear(x, weight)


class CompactTokenEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if embedding_dim <= 4:
            raise ValueError("embedding_dim must exceed the four-channel bottleneck")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compact_dim = embedding_dim - 4
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, self.compact_dim)
        )

        # Retain the former full embedding's conceptual initialization so this
        # change isolates representation width and preserves the random stream.
        size = num_embeddings * embedding_dim
        inv_sqrt = size ** -0.5
        reflector = torch.full((size,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("init_reflector", reflector, persistent=False)
        self.init_reflector_norm_sq = float(reflector.dot(reflector))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            size = self.num_embeddings * self.embedding_dim
            conceptual_coordinates = self.weight.new_empty(size - 1)
            nn.init.normal_(conceptual_coordinates, mean=0.0, std=0.02)
            flat = F.pad(conceptual_coordinates, (1, 0))
            projection = (
                2.0
                * torch.dot(self.init_reflector, flat)
                / self.init_reflector_norm_sq
            )
            flat = flat - self.init_reflector * projection
            conceptual_weight = flat.view(
                self.num_embeddings, self.embedding_dim
            )
            self.weight.copy_(conceptual_weight[:, : self.compact_dim])

    def materialized_weight(self) -> torch.Tensor:
        return F.pad(
            self.weight, (0, self.embedding_dim - self.compact_dim)
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = OneRowInputGaugeLinear(d_model, 3 * d_model)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        # A constant shift of every lag bias cancels in the softmax, so lag
        # zero is the fixed reference coordinate.
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
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
        positions = torch.arange(seqlen, device=x.device)
        lags = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias = F.pad(self.relative_bias_coordinates, (1, 0))
        att = att + relative_bias[:, lags].unsqueeze(0)
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
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
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
        self.token_emb = CompactTokenEmbedding(cfg.vocab_size, cfg.d_model)

        # Preserve the constructor-time random draw of the removed Embedding
        # so all downstream modules retain their verified initialization.
        conceptual_position = torch.empty(cfg.max_seq_len, cfg.d_model)
        nn.init.normal_(conceptual_position)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Reproduce the original apply-time order, including the removed
        # position table's draw, before initializing the transformer block.
        self.token_emb.reset_parameters()
        nn.init.normal_(conceptual_position, mean=0.0, std=0.02)
        self.blocks.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, CompactTokenEmbedding):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
        elif isinstance(module, OneColumnShiftGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, OneRowInputGaugeLinear):
            module.reset_parameters()
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

        x = self.drop(self.token_emb(idx))

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
