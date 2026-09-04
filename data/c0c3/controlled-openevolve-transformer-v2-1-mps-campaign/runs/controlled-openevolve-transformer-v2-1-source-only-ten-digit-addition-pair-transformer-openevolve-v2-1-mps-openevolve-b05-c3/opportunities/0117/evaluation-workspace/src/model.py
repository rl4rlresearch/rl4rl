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


class GaugeFixedEmbedding(nn.Embedding):
    """Tied embedding modulo one hidden-coordinate shift per token."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 1)
        )

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.num_embeddings, 1),
            ),
            dim=-1,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm whose sole adaptive scale is shared with final LayerNorm."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model - 1))
        object.__setattr__(self, "_scale_source", None)

    def share_scale(self, source: nn.Module) -> None:
        object.__setattr__(self, "_scale_source", source)

    def full_weight(self) -> torch.Tensor:
        source_weight = self._scale_source.full_weight()
        shared_scale = source_weight.mean().reshape(1)
        return torch.cat((shared_scale, self.fixed_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and fully tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with all scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model))

    def full_weight(self) -> torch.Tensor:
        return self.fixed_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class GaugeFixedValueLinear(nn.Linear):
    """Value map modulo its LayerNorm-nullspace row shifts."""

    def __init__(self, d_model: int):
        # Preserve the original QKV constructor's RNG consumption.
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.out_features = d_model
        self.weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.weight, self.weight.new_zeros(self.d_model, 1)),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight())


class GaugeFixedFC1Linear(nn.Linear):
    """MLP input map modulo its LayerNorm-nullspace row shifts."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_model, d_ff)
        self.d_model = d_model
        self.d_ff = d_ff
        self.weight = nn.Parameter(torch.empty(d_ff, d_model - 1))

    def full_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.weight, self.weight.new_zeros(self.d_ff, 1)),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.full_weight(), self.bias)


class GaugeTiedHead(nn.Linear):
    """Parameter-free view of the globally gauge-fixed tied embedding."""

    def __init__(self, embedding: GaugeFixedEmbedding):
        # Preserve the original tied Linear constructor's RNG consumption.
        super().__init__(
            embedding.embedding_dim,
            embedding.num_embeddings,
            bias=False,
        )
        self.weight = None
        object.__setattr__(self, "_embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._embedding.full_weight())


class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with all common-output shifts fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_start = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty((d_model - 1) * d_model))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)


class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with all common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = (d_model - 1) * d_ff
        self.weight = nn.Parameter(torch.empty((d_model - 1) * d_ff))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.d_ff),
            )
        )
        return flat.view(self.d_model, self.d_ff)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.value = GaugeFixedValueLinear(d_model)
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the four terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 4)
        )
        self.register_buffer(
            "relative_shift",
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head),
        )

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        v = self.value(x)
        v = v.view(
            bsz, seqlen, self.n_head, self.head_dim
        ).transpose(1, 2)
        positions = torch.arange(seqlen, device=x.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(4),
            )
        )
        frequency = torch.arange(
            base_bias.numel() // 2 + 1,
            device=x.device,
            dtype=base_bias.dtype,
        )
        head_shift = torch.cat(
            (
                self.relative_shift.new_zeros(1),
                self.relative_shift,
            )
        )
        phase = torch.exp(
            -2j
            * math.pi
            * head_shift[:, None]
            * frequency[None, :]
            / base_bias.numel()
        )
        lag_bias = torch.fft.irfft(
            torch.fft.rfft(base_bias).unsqueeze(0) * phase,
            n=base_bias.numel(),
            dim=-1,
        )
        lag_bias = lag_bias - lag_bias[:, -1:]
        att = lag_bias[:, lag].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        output_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        y = F.linear(y, self.proj.full_weight(), output_bias)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = GaugeFixedFC1Linear(d_model, d_ff)
        self.fc2 = GaugeFixedMLPProjectionLinear(d_model, d_ff)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 2))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(2))
        )
        output = F.linear(
            F.gelu(hidden), self.fc2.full_weight(), output_bias
        )
        return self.drop(output)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = GaugeFixedScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = GaugeFixedMLPScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = GaugeFixedEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.ln_f = TiedFinalScaleLayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        for block in self.blocks:
            block.ln1.share_scale(self.ln_f)

        # Weight tying with the reconstructed full input embedding.
        self.lm_head = GaugeTiedHead(self.token_emb)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                module.weight.copy_(full[:, :-1])
        elif isinstance(module, GaugeFixedValueLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                value = full[2 * d_model :]
                omitted = value[:, -1:].clone()
                value[:, :-1].sub_(omitted)
                value[:, -1].zero_()
                module.weight.copy_(value[:, :-1])
        elif isinstance(module, GaugeFixedFC1Linear):
            full = module.weight.new_empty(
                module.d_ff, module.d_model
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                module.weight.copy_(full[:, :-1])
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeTiedHead):
            embedding = module._embedding
            full = embedding.weight.new_empty(
                embedding.num_embeddings, embedding.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[:, -1:].clone()
                full[:, :-1].sub_(omitted)
                full[:, -1].zero_()
                embedding.weight.copy_(full[:, :-1])
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                module.weight.copy_(full[:-1].reshape(-1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                module.weight.copy_(full[:-1].reshape(-1))
                if module.bias is not None:
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

        x = self.drop(self.token_emb(idx))

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
