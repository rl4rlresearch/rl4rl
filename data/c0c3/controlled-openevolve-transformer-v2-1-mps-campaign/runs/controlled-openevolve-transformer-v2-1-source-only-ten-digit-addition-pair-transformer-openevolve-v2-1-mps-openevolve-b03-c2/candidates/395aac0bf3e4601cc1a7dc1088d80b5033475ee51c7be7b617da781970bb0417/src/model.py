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
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Within each head, an invertible query/key basis change
        # can additionally fix one generic query-bias coordinate to zero.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Downstream LayerNorms cancel the uniform output coordinate. Two
        # shears are fixed in each head; scalar value/output gauges normalize
        # both target rows and one additional row in the first head.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj_first_weight = nn.Parameter(
            torch.empty(d_model - 2)
        )
        self.proj_head_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.proj_last_weight = nn.Parameter(
            torch.empty(d_model - 4)
        )
        self.register_buffer(
            "proj_first_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_head_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "proj_last_pivot",
            torch.zeros((), dtype=torch.long),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
        value_bias = self.qkv.bias.new_zeros(d_model)
        qkv_bias = torch.cat(
            (
                query_bias,
                query_bias.mean().expand(d_model),
                value_bias,
            )
        )
        qkv_weight_relative = torch.cat(
            (
                self.qkv.weight,
                self.qkv.weight.new_zeros(
                    (self.qkv.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        qkv_weight = (
            qkv_weight_relative
            + self.qkv.weight.mean(dim=-1, keepdim=True)
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
        first_pivot = int(self.proj_first_pivot.item())
        first_chart = torch.cat(
            (
                self.proj_first_weight[:first_pivot],
                self.proj_first_weight.new_full((1,), 1.0),
                self.proj_first_weight[first_pivot:],
            )
        )
        first_relative = first_chart * (
            (0.02 * math.sqrt(d_model - 1))
            / first_chart.norm()
        )
        head_pivot = int(self.proj_head_pivot.item())
        head_chart = torch.cat(
            (
                self.proj_head_weight[:head_pivot],
                self.proj_head_weight.new_full((1,), 1.0),
                self.proj_head_weight[head_pivot:],
            )
        )
        head_free = head_chart * (
            (0.02 * math.sqrt(d_model - 3))
            / head_chart.norm()
        )
        last_pivot = int(self.proj_last_pivot.item())
        last_chart = torch.cat(
            (
                self.proj_last_weight[:last_pivot],
                self.proj_last_weight.new_full((1,), 1.0),
                self.proj_last_weight[last_pivot:],
            )
        )
        last_free = last_chart * (
            (0.02 * math.sqrt(d_model - 3))
            / last_chart.norm()
        )
        head_relative = torch.cat(
            (head_free, head_free.new_zeros(2))
        )
        last_relative = torch.cat(
            (last_free, last_free.new_zeros(2))
        )
        split = self.head_dim - 2
        weight_rows = torch.cat(
            (
                first_relative.unsqueeze(0),
                self.proj.weight[:split],
                head_relative.unsqueeze(0),
                self.proj.weight[split:],
                last_relative.unsqueeze(0),
            ),
            dim=0,
        )
        weight_relative = torch.cat(
            (
                weight_rows,
                weight_rows.new_zeros((weight_rows.size(0), 1)),
            ),
            dim=-1,
        )
        proj_weight = (
            weight_relative
            + weight_rows.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # The final LayerNorm cancels the feature-uniform component of this
        # residual bias, so retain only its seven relative coordinates.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight_relative = torch.cat(
            (
                self.fc1.weight,
                self.fc1.weight.new_zeros(
                    (self.fc1.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc1_weight = (
            fc1_weight_relative
            + self.fc1.weight.mean(dim=-1, keepdim=True)
        )
        hidden = F.gelu(F.linear(x, fc1_weight, self.fc1.bias))
        weight_relative = torch.cat(
            (
                self.fc2.weight,
                self.fc2.weight.new_zeros(
                    (self.fc2.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc2_weight = (
            weight_relative
            + self.fc2.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, fc2_weight, fc2_bias))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Learned scales are folded into their downstream weights by the
        # factor-aware optimizer.
        self.ln1 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
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
        # A common positive scale of this affine transform changes only the
        # global logit temperature, not autoregressive argmax decoding.
        self.ln_f = nn.LayerNorm(
            cfg.d_model, elementwise_affine=False
        )
        self.ln_f_weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.ln_f_bias = nn.Parameter(torch.zeros(cfg.d_model))

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Store LayerNorm-null directions through relative representatives.
        # Two conditioned shears are fixed per head, then diagonal gauges
        # normalize both target rows and one additional first-head row.
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach().clone()
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1).clone()
            )
            relative_proj_weight = (
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            value_start = 2 * cfg.d_model
            target_rows = (
                block.attn.head_dim - 1,
                cfg.d_model - 1,
            )

            for target_row in target_rows:
                head_start = (
                    target_row - block.attn.head_dim + 1
                )
                pivot_pairs = [
                    (left, right)
                    for left in range(head_start, target_row)
                    for right in range(left + 1, target_row)
                ]
                pivots = max(
                    pivot_pairs,
                    key=lambda pair: abs(
                        torch.linalg.det(
                            relative_proj_weight[list(pair), -2:]
                        ).item()
                    ),
                )
                matrix = relative_proj_weight[
                    list(pivots), -2:
                ].transpose(0, 1)
                coefficients = torch.linalg.solve(
                    matrix,
                    relative_proj_weight[target_row, -2:],
                )
                relative_proj_weight[target_row] = (
                    relative_proj_weight[target_row]
                    - (
                        coefficients.unsqueeze(1)
                        * relative_proj_weight[list(pivots)]
                    ).sum(dim=0)
                )
                for pivot, coefficient in zip(
                    pivots, coefficients
                ):
                    full_qkv_weight[value_start + pivot] = (
                        full_qkv_weight[value_start + pivot]
                        + coefficient
                        * full_qkv_weight[value_start + target_row]
                    )

            first_row = 0
            first_free = relative_proj_weight[first_row]
            first_pivot = int(first_free.abs().argmax().item())
            first_pivot_value = first_free[first_pivot]
            first_chart = first_free / first_pivot_value
            first_gauge_norm = 0.02 * math.sqrt(first_free.numel())
            first_scale = (
                first_pivot_value.sign()
                * first_free.norm()
                / first_gauge_norm
            )
            full_qkv_weight[value_start + first_row] = (
                first_scale
                * full_qkv_weight[value_start + first_row]
            )
            block.attn.proj_first_pivot.fill_(first_pivot)

            first_target = block.attn.head_dim - 1
            head_free = relative_proj_weight[first_target, :-2]
            head_pivot = int(head_free.abs().argmax().item())
            head_pivot_value = head_free[head_pivot]
            head_chart = head_free / head_pivot_value
            head_gauge_norm = 0.02 * math.sqrt(head_free.numel())
            head_scale = (
                head_pivot_value.sign()
                * head_free.norm()
                / head_gauge_norm
            )
            full_qkv_weight[value_start + first_target] = (
                head_scale
                * full_qkv_weight[value_start + first_target]
            )
            block.attn.proj_head_pivot.fill_(head_pivot)

            last_target = cfg.d_model - 1
            last_free = relative_proj_weight[last_target, :-2]
            last_pivot = int(last_free.abs().argmax().item())
            last_pivot_value = last_free[last_pivot]
            last_chart = last_free / last_pivot_value
            last_gauge_norm = 0.02 * math.sqrt(last_free.numel())
            last_scale = (
                last_pivot_value.sign()
                * last_free.norm()
                / last_gauge_norm
            )
            full_qkv_weight[value_start + last_target] = (
                last_scale
                * full_qkv_weight[value_start + last_target]
            )
            block.attn.proj_last_pivot.fill_(last_pivot)

            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            block.attn.proj.weight = nn.Parameter(
                torch.cat(
                    (
                        relative_proj_weight[
                            first_row + 1:first_target
                        ],
                        relative_proj_weight[
                            first_target + 1:-1
                        ],
                    ),
                    dim=0,
                )
            )
            block.attn.proj_first_weight = nn.Parameter(
                torch.cat(
                    (
                        first_chart[:first_pivot],
                        first_chart[first_pivot + 1:],
                    )
                )
            )
            block.attn.proj_head_weight = nn.Parameter(
                torch.cat(
                    (
                        head_chart[:head_pivot],
                        head_chart[head_pivot + 1:],
                    )
                )
            )
            block.attn.proj_last_weight = nn.Parameter(
                torch.cat(
                    (
                        last_chart[:last_pivot],
                        last_chart[last_pivot + 1:],
                    )
                )
            )

            full_fc1_weight = block.mlp.fc1.weight.detach()
            block.mlp.fc1.weight = nn.Parameter(
                full_fc1_weight[:, :-1] - full_fc1_weight[:, -1:]
            )
            full_fc2_weight = (
                block.mlp.fc2.weight.detach().transpose(0, 1)
            )
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )

        # Shifting every token row by the same feature vector and every
        # positional row by its negative leaves all inputs unchanged and
        # changes output logits only by a vocabulary-uniform scalar. Fix the
        # final token row to zero and transfer its offset into positions.
        full_token_weight = self.token_emb.weight.detach()
        token_offset = full_token_weight[-1:]
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - token_offset
        )
        self.lm_head.weight = self.token_emb.weight

        # Positional rows also retain only coordinates relative to their
        # final feature, which downstream LayerNorms cancel.
        full_pos_weight = self.pos_emb.weight.detach() + token_offset
        self.pos_emb.weight = nn.Parameter(
            full_pos_weight[:, :-1] - full_pos_weight[:, -1:]
        )

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

        token_weight = torch.cat(
            (
                self.token_emb.weight,
                self.token_emb.weight.new_zeros(
                    (1, self.cfg.d_model)
                ),
            ),
            dim=0,
        )

        pos_relative = torch.cat(
            (
                self.pos_emb.weight,
                self.pos_emb.weight.new_zeros(
                    (self.pos_emb.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        pos_weight = pos_relative + self.pos_emb.weight.mean(
            dim=-1, keepdim=True
        )

        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        ln_f_weight = torch.cat(
            (
                self.ln_f_weight,
                self.ln_f_weight.new_ones(1),
            )
        )
        x = x * ln_f_weight + self.ln_f_bias
        logits = F.linear(x, token_weight)

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
