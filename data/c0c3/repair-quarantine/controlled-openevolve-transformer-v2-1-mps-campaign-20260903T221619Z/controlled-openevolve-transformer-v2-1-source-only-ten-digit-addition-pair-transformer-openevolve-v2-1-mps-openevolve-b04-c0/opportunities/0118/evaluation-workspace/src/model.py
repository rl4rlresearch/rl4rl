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


class SharedAnchorEmbeddings(nn.Module):
    """Token and position tables sharing eight gauge-redundant scalars."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.anchor = nn.Parameter(torch.empty(8))
        self.token_rest = nn.Parameter(torch.empty(vocab_size * d_model - 8))
        self.pos_rest = nn.Parameter(torch.empty(max_seq_len * d_model - 8))

        # Match the random-number consumption of two nn.Embedding constructors.
        nn.init.normal_(torch.empty(vocab_size, d_model))
        nn.init.normal_(torch.empty(max_seq_len, d_model))

    def token_weight(self) -> torch.Tensor:
        fourth_index = self.d_model + 3
        fifth_index = 2 * self.d_model + 4
        sixth_index = 3 * self.d_model + 5
        seventh_index = 4 * self.d_model + 6
        eighth_index = 5 * self.d_model + 7
        fourth_offset = fourth_index - 3
        fifth_offset = fourth_offset + fifth_index - fourth_index - 1
        sixth_offset = fifth_offset + sixth_index - fifth_index - 1
        seventh_offset = sixth_offset + seventh_index - sixth_index - 1
        eighth_offset = seventh_offset + eighth_index - seventh_index - 1
        flat = torch.cat(
            (
                self.anchor[:3],
                self.token_rest[:fourth_offset],
                self.anchor[3:4],
                self.token_rest[fourth_offset:fifth_offset],
                self.anchor[4:5],
                self.token_rest[fifth_offset:sixth_offset],
                self.anchor[5:6],
                self.token_rest[sixth_offset:seventh_offset],
                self.anchor[6:7],
                self.token_rest[seventh_offset:eighth_offset],
                self.anchor[7:],
                self.token_rest[eighth_offset:],
            )
        )
        return flat.view(self.vocab_size, self.d_model)

    def pos_weight(self) -> torch.Tensor:
        # The third through eighth anchors train only through their token uses,
        # avoiding additional positional-gradient coupling.
        pos_anchor = torch.cat((self.anchor[:2], self.anchor[2:].detach()))
        return torch.cat((pos_anchor, self.pos_rest)).view(self.max_seq_len, self.d_model)

    def token(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.token_weight())

    def position(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.pos_weight())


class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with six zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 13))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        fixed = self.weight_rest.new_full((1,), 0.02)
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                fixed,
                self.weight_rest[2:5],
                fixed,
                self.weight_rest[5:12],
                fixed,
                self.weight_rest[12:15],
                zero,
                self.weight_rest[15:18],
                zero,
                self.weight_rest[18:25],
                fixed,
                self.weight_rest[25:32],
                fixed,
                self.weight_rest[32:39],
                fixed,
                self.weight_rest[39:46],
                fixed,
                self.weight_rest[46:56],
                zero,
                self.weight_rest[56:87],
                zero,
                self.weight_rest[87:],
            )
        ).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor())


class ResidualGaugeLinear(nn.Module):
    """Projection with two weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 2)
        )
        self.bias = None

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (zero, self.weight_rest[:3], zero, self.weight_rest[3:])
        ).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_tensor(), self.bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = AttentionWeightAnchoredLinear(d_model, 3 * d_model)
        self.q_bias_rest = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = ResidualGaugeLinear(d_model, d_model)
        self.proj.bias = self.v_bias
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = torch.cat((self.q_bias_rest.new_zeros(3), self.q_bias_rest))
        q = q + q_bias
        v = v + self.v_bias

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


class BiasAnchoredLinear(nn.Linear):
    """Linear layer with only bias coordinate 8 learned."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 11))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat((zeros(8), self.bias_rest, zeros(3)))
        return F.linear(x, self.weight, bias)


class FinalBiasAnchoredLinear(nn.Module):
    """MLP output projection with one residual-gauge weight and bias anchor."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((self.bias_rest[:1], zero, self.bias_rest[1:]))
        return F.linear(x, self.weight_tensor(), bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = BiasAnchoredLinear(d_model, d_ff)
        self.fc2 = FinalBiasAnchoredLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class AttentionAnchoredLayerNorm(nn.Module):
    """LayerNorm with gain 3 fixed and gain 7 sharing gain 1."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 7))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        one = self.weight_rest.new_ones(1)
        weight = torch.cat(
            (
                one,
                self.weight_rest[:1],
                one,
                one,
                one,
                one,
                one,
                self.weight_rest[:1].detach(),
            )
        )
        bias = torch.cat((self.bias_rest[:3], zero, self.bias_rest[3:]))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with only gain 1 learned and all other gains absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 7))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((ones(1), self.weight_rest, ones(6)))
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(2), self.bias_rest[1:], zeros(2))
        )
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = AttentionAnchoredLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = MLPAnchoredLayerNorm(cfg.d_model)

        # The first MLP gain and the remaining attention gain are independent
        # downstream-absorbable gauges, so learn one scalar for both.
        self.ln1.weight_rest = self.ln2.weight_rest

        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = SharedAnchorEmbeddings(cfg.vocab_size, cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, AttentionWeightAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                weight[0, 0] = 0.0
                weight[0, 2] = 0.0

                # Fix two nonzero query coefficients using independent Q/K
                # reciprocal scales while preserving the attention scores.
                scale = 0.02 / weight[0, 4]
                weight[0].mul_(scale)
                weight[8].div_(scale)

                scale = 0.02 / weight[1, 0]
                weight[1].mul_(scale)
                weight[9].div_(scale)

                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                # Use query row 0's stable fixed coefficient to eliminate
                # qkv.weight[2, 4]. The reciprocal key shear preserves every
                # head-0 attention score and leaves row 2's scale anchor fixed.
                shear = weight[2, 4] / weight[0, 4]
                weight[2].sub_(shear * weight[0])
                weight[8].add_(shear * weight[10])
                weight[2, 4] = 0.0

                # Canonicalize the corresponding component-zero key row in
                # head 0. At initialization LayerNorm has unit gain, so this
                # only adds a softmax-invariant constant to every key.
                key_offset = weight[8, 3].clone()
                weight[8].sub_(key_offset)
                weight[8, 3] = 0.0

                scale = 0.02 / weight[4, 0]
                weight[4].mul_(scale)
                weight[12].div_(scale)

                scale = 0.02 / weight[5, 0]
                weight[5].mul_(scale)
                weight[13].div_(scale)

                scale = 0.02 / weight[6, 0]
                weight[6].mul_(scale)
                weight[14].div_(scale)

                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                # Canonicalize one key row along LayerNorm's null direction.
                # The resulting position-independent key shift cancels in
                # every attention softmax.
                key_offset = weight[12, 3].clone()
                weight[12].sub_(key_offset)
                weight[12, 3] = 0.0

                # Use query row 1 as a stable nonzero pivot to eliminate
                # qkv.weight[3, 0]. Apply the inverse shear to its matching
                # key row so every head-0 attention score is preserved.
                shear = weight[3, 0] / weight[1, 0]
                weight[3].sub_(shear * weight[1])
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:20],
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:67],
                            flat[68:99],
                            flat[100:],
                        )
                    )
                )
        elif isinstance(module, ResidualGaugeLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Subtracting either input-column coefficient from every
                # output row adds only a feature-uniform residual shift.
                # Subsequent LayerNorms remove both shifts exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:4], flat[5:]))
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, FinalBiasAnchoredLinear):
            weight = module.weight_rest.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(weight, mean=0.0, std=0.02)
            with torch.no_grad():
                # Removing this coefficient from every output row changes the
                # residual stream only by a feature-uniform, tokenwise shift.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0
                module.weight_rest.copy_(weight.flatten()[1:])
                nn.init.zeros_(module.bias_rest)
        elif isinstance(module, SharedAnchorEmbeddings):
            token = module.token_rest.new_empty(module.vocab_size, module.d_model)
            pos = module.pos_rest.new_empty(module.max_seq_len, module.d_model)
            nn.init.normal_(token, mean=0.0, std=0.02)
            nn.init.normal_(pos, mean=0.0, std=0.02)

            # Gauge-transform the ordinary initialization so each anchor pair
            # agrees while token-plus-position inputs remain unchanged.
            shift = 0.5 * (pos[0, :3] - token[0, :3])
            token[:, :3].add_(shift)
            pos[:, :3].sub_(shift)

            fourth_shift = 0.5 * (pos[0, 3] - token[1, 3])
            token[:, 3].add_(fourth_shift)
            pos[:, 3].sub_(fourth_shift)

            fifth_shift = 0.5 * (pos[0, 4] - token[2, 4])
            token[:, 4].add_(fifth_shift)
            pos[:, 4].sub_(fifth_shift)

            sixth_shift = 0.5 * (pos[0, 5] - token[3, 5])
            token[:, 5].add_(sixth_shift)
            pos[:, 5].sub_(sixth_shift)

            seventh_shift = 0.5 * (pos[0, 6] - token[4, 6])
            token[:, 6].add_(seventh_shift)
            pos[:, 6].sub_(seventh_shift)

            eighth_shift = 0.5 * (pos[0, 7] - token[5, 7])
            token[:, 7].add_(eighth_shift)
            pos[:, 7].sub_(eighth_shift)

            fourth_index = module.d_model + 3
            fifth_index = 2 * module.d_model + 4
            sixth_index = 3 * module.d_model + 5
            seventh_index = 4 * module.d_model + 6
            eighth_index = 5 * module.d_model + 7
            token_flat = token.flatten()
            with torch.no_grad():
                module.anchor.copy_(
                    torch.cat(
                        (
                            token[0, :3],
                            token[1, 3:4],
                            token[2, 4:5],
                            token[3, 5:6],
                            token[4, 6:7],
                            token[5, 7:8],
                        )
                    )
                )
                module.token_rest.copy_(
                    torch.cat(
                        (
                            token_flat[3:fourth_index],
                            token_flat[fourth_index + 1 : fifth_index],
                            token_flat[fifth_index + 1 : sixth_index],
                            token_flat[sixth_index + 1 : seventh_index],
                            token_flat[seventh_index + 1 : eighth_index],
                            token_flat[eighth_index + 1 :],
                        )
                    )
                )
                module.pos_rest.copy_(pos.flatten()[8:])
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
        x = self.embeddings.token(idx) + self.embeddings.position(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = F.linear(x, self.embeddings.token_weight())

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
