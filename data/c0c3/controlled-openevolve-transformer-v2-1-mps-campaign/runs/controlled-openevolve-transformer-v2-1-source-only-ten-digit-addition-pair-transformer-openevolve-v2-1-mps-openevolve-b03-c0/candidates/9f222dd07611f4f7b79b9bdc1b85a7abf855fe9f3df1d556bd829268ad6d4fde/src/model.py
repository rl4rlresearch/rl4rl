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
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 16))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.weight = nn.Parameter(torch.empty(d_model, d_model - 2))
        self.proj_col_2 = nn.Parameter(torch.empty(d_model - 1))
        self.proj_col = nn.Parameter(torch.empty(d_model - 1))
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias[d_model - 1 : d_model].detach(),
                self.qkv.bias.new_zeros(d_model),
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
        proj_col_2 = torch.cat(
            (self.proj_col_2, self.proj_col_2.new_zeros(1))
        ).unsqueeze(1)
        proj_col = torch.cat(
            (self.proj_col, self.proj_col.new_zeros(1))
        ).unsqueeze(1)
        proj_weight = torch.cat(
            (self.proj.weight, proj_col_2, proj_col), dim=1
        )
        proj_bias = torch.cat((self.proj.bias, self.proj.bias.new_zeros(1)))
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2._defer_two_column_gauge = True
        self.fc2.register_parameter("unused", None)
        self.register_parameter("fc2_col_5", None)
        self.register_parameter("fc2_col_4", None)
        self.register_parameter("fc2_col_3", None)
        self.register_parameter("fc2_col_2", None)
        self.register_parameter("fc2_col", None)
        self.register_parameter("fc2_col_0", None)
        self.register_parameter("fc2_col_1", None)
        self.register_parameter("fc2_col_low", None)
        self.register_parameter("fc2_col_mid", None)
        self.register_parameter("fc2_col_4_abs", None)
        self.register_parameter("fc2_col_5_abs", None)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = torch.cat(
            (self.fc1.weight, self.fc1.weight.new_zeros(1))
        ).view(self.fc1.out_features, self.fc1.in_features)
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_col_0 = torch.cat(
            (self.fc2_col_0, self.fc2_col_0.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_1 = torch.cat(
            (self.fc2_col_1, self.fc2_col_1.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_low = torch.cat(
            (self.fc2_col_low, self.fc2_col_low.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_mid = torch.cat(
            (self.fc2_col_mid, self.fc2_col_mid.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_4_abs = torch.cat(
            (self.fc2_col_4_abs, self.fc2_col_4_abs.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_5_abs = torch.cat(
            (self.fc2_col_5_abs, self.fc2_col_5_abs.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_5 = torch.cat(
            (self.fc2_col_5, self.fc2_col_5.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_4 = torch.cat(
            (self.fc2_col_4, self.fc2_col_4.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_3 = torch.cat(
            (self.fc2_col_3, self.fc2_col_3.new_zeros(1))
        ).unsqueeze(1)
        fc2_col_2 = torch.cat(
            (self.fc2_col_2, self.fc2_col_2.new_zeros(1))
        ).unsqueeze(1)
        fc2_col = torch.cat(
            (self.fc2_col, self.fc2_col.new_zeros(1))
        ).unsqueeze(1)
        fc2_weight = torch.cat(
            (
                fc2_col_0,
                fc2_col_1,
                fc2_col_low,
                fc2_col_mid,
                fc2_col_4_abs,
                fc2_col_5_abs,
                self.fc2.weight,
                fc2_col_5,
                fc2_col_4,
                fc2_col_3,
                fc2_col_2,
                fc2_col,
            ),
            dim=1,
        )
        fc2_bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.weight = self.ln1.weight
        self.ln2.bias = self.ln1.bias
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
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * cfg.d_model - 1)
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_antepenultimate = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_penultimate = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_last = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.ln_f.bias = self.blocks[0].ln1.bias

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        with torch.no_grad():
            for block in self.blocks:
                for proj_col in (block.attn.proj_col_2, block.attn.proj_col):
                    full_proj_col = proj_col.new_empty(cfg.d_model)
                    nn.init.normal_(full_proj_col, mean=0.0, std=0.02)
                    full_proj_col.sub_(full_proj_col[-1].clone())
                    proj_col.copy_(full_proj_col[:-1])

                full_fc2_col = block.mlp.fc2.weight.new_empty(cfg.d_model)
                nn.init.normal_(full_fc2_col, mean=0.0, std=0.02)
                full_fc2_col_0 = block.mlp.fc2.weight[:, 0].detach().clone()
                full_fc2_col_1 = block.mlp.fc2.weight[:, 1].detach().clone()
                full_fc2_col_low = block.mlp.fc2.weight[:, 2].detach().clone()
                full_fc2_col_mid = block.mlp.fc2.weight[:, 3].detach().clone()
                full_fc2_col_4_abs = block.mlp.fc2.weight[:, 4].detach().clone()
                full_fc2_col_5_abs = block.mlp.fc2.weight[:, 5].detach().clone()
                full_fc2_col_5 = block.mlp.fc2.weight[:, -5].detach().clone()
                full_fc2_col_4 = block.mlp.fc2.weight[:, -4].detach().clone()
                full_fc2_col_3 = block.mlp.fc2.weight[:, -3].detach().clone()
                full_fc2_col_2 = block.mlp.fc2.weight[:, -2].detach().clone()
                base_fc2_weight = block.mlp.fc2.weight[:, 6:-5].detach().clone()
                block.mlp.fc2.weight = nn.Parameter(base_fc2_weight)

                full_fc2_col_0.sub_(full_fc2_col_0[-1].clone())
                block.mlp.fc2_col_0 = nn.Parameter(
                    full_fc2_col_0[:-1].clone()
                )
                full_fc2_col_1.sub_(full_fc2_col_1[-1].clone())
                block.mlp.fc2_col_1 = nn.Parameter(
                    full_fc2_col_1[:-1].clone()
                )
                full_fc2_col_low.sub_(full_fc2_col_low[-1].clone())
                block.mlp.fc2_col_low = nn.Parameter(
                    full_fc2_col_low[:-1].clone()
                )
                full_fc2_col_mid.sub_(full_fc2_col_mid[-1].clone())
                block.mlp.fc2_col_mid = nn.Parameter(
                    full_fc2_col_mid[:-1].clone()
                )
                full_fc2_col_4_abs.sub_(full_fc2_col_4_abs[-1].clone())
                block.mlp.fc2_col_4_abs = nn.Parameter(
                    full_fc2_col_4_abs[:-1].clone()
                )
                full_fc2_col_5_abs.sub_(full_fc2_col_5_abs[-1].clone())
                block.mlp.fc2_col_5_abs = nn.Parameter(
                    full_fc2_col_5_abs[:-1].clone()
                )
                full_fc2_col_5.sub_(full_fc2_col_5[-1].clone())
                block.mlp.fc2_col_5 = nn.Parameter(
                    full_fc2_col_5[:-1].clone()
                )
                full_fc2_col_4.sub_(full_fc2_col_4[-1].clone())
                block.mlp.fc2_col_4 = nn.Parameter(
                    full_fc2_col_4[:-1].clone()
                )
                full_fc2_col_3.sub_(full_fc2_col_3[-1].clone())
                block.mlp.fc2_col_3 = nn.Parameter(
                    full_fc2_col_3[:-1].clone()
                )
                full_fc2_col_2.sub_(full_fc2_col_2[-1].clone())
                block.mlp.fc2_col_2 = nn.Parameter(
                    full_fc2_col_2[:-1].clone()
                )
                full_fc2_col.sub_(full_fc2_col[-1].clone())
                block.mlp.fc2_col = nn.Parameter(full_fc2_col[:-1].clone())

            full_token_weight = self.token_emb.weight.new_empty(
                cfg.vocab_size * cfg.d_model
            )
            nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
            full_token_weight.sub_(full_token_weight[-1].clone())
            self.token_emb.weight.copy_(full_token_weight[:-1])

            full_pos_antepenultimate = self.pos_emb.weight[-3].detach().clone()
            full_pos_penultimate = self.pos_emb.weight[-2].detach().clone()
            full_pos_row = self.pos_emb.weight[-1].detach().clone()
            self.pos_emb.weight = nn.Parameter(
                self.pos_emb.weight[:-3].detach().clone()
            )
            full_pos_antepenultimate.sub_(
                full_pos_antepenultimate[-1].clone()
            )
            self.pos_emb_antepenultimate.copy_(
                full_pos_antepenultimate[:-1]
            )
            full_pos_penultimate.sub_(full_pos_penultimate[-1].clone())
            self.pos_emb_penultimate.copy_(full_pos_penultimate[:-1])
            full_pos_row.sub_(full_pos_row[-1].clone())
            self.pos_emb_last.copy_(full_pos_row[:-1])

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_defer_two_column_gauge", False):
                initialized_weight = module.weight.new_empty(
                    module.out_features, module.in_features - 1
                )
                nn.init.normal_(initialized_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight[:, :-1].copy_(initialized_weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _token_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.token_emb.weight, self.token_emb.weight.new_zeros(1))
        ).view(self.cfg.vocab_size, self.cfg.d_model)

    def _position_weight(self) -> torch.Tensor:
        antepenultimate_row = torch.cat(
            (
                self.pos_emb_antepenultimate,
                self.pos_emb_antepenultimate.new_zeros(1),
            )
        ).unsqueeze(0)
        penultimate_row = torch.cat(
            (
                self.pos_emb_penultimate,
                self.pos_emb_penultimate.new_zeros(1),
            )
        ).unsqueeze(0)
        last_row = torch.cat(
            (self.pos_emb_last, self.pos_emb_last.new_zeros(1))
        ).unsqueeze(0)
        return torch.cat(
            (
                self.pos_emb.weight,
                antepenultimate_row,
                penultimate_row,
                last_row,
            ),
            dim=0,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, seqlen = idx.shape

        if seqlen > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.cfg.max_seq_len :]
            seqlen = idx.shape[1]

        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_weight = self._token_weight()
        position_weight = self._position_weight()
        x = F.embedding(idx, token_weight) + F.embedding(pos, position_weight)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
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
