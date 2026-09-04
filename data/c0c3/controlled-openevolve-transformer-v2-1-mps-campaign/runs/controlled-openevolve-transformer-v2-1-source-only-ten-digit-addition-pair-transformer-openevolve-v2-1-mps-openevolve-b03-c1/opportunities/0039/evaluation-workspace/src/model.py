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
        # Construct the original affine first to preserve the baseline RNG
        # stream. The Block supplies normalized inputs directly, so retain a
        # full direct query bias while storing all three input projections in
        # the observable zero-mean LayerNorm subspace.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.proj.bias = nn.Parameter(self.proj.bias[:-1].clone())
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Any all-ones component emitted into the residual stream is removed
        # by subsequent LayerNorms. Parameterize the observable zero-mean
        # output subspace with an orthonormal basis.
        basis = torch.eye(d_model)[:, : d_model - 1]
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.register_buffer("proj_basis", basis, persistent=False)

        # Distance zero fixes each head's softmax-invariant additive gauge;
        # only relative differences between its distance biases are learned.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def gauge_fix_qkv(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.qkv.weight
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_coeff.reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )

    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        matrix_size = d_model * (d_model - 1)
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_coeff = self.qkv.weight[matrix_size : 2 * matrix_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            d_model, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight, self.qkv.bias)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias = torch.cat(
            (self.relative_bias.new_zeros(self.n_head, 1), self.relative_bias),
            dim=1,
        )
        att = att + relative_bias[:, distance].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_weight = self.proj_basis @ self.proj.weight
        proj_bias = self.proj_basis @ self.proj.bias
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

        # Non-affine LayerNorm produces zero-mean inputs, so the all-ones
        # component of every fc1 row is unobservable. Build an orthonormal
        # basis for the complementary zero-mean subspace.
        basis = torch.eye(d_model)[:, : d_model - 1]
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.register_buffer("fc1_basis", basis.T, persistent=False)

    def gauge_fix_fc1(self) -> None:
        with torch.no_grad():
            weight = self.fc1.weight
            centered = weight - weight.mean(dim=1, keepdim=True)
            self.fc1.weight = nn.Parameter(
                (centered @ self.fc1_basis.T).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        return self.drop(self.fc2(F.gelu(hidden)))


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
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Compress only after full initialization, preserving the original RNG
        # stream and the initialized function on LayerNorm outputs.
        for block in self.blocks:
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()

        # Preserve the baseline RNG stream by initializing the original table.
        # The transient factorization below keeps initialization behavior aligned
        # with the prior design before fixed harmonic coordinates replace it.
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(
                self.pos_emb.weight, full_matrices=False
            )
            scale = singular[:4].sqrt()
            pos_code = left[:, :4] * scale
            pos_proj = scale.unsqueeze(1) * right[:4]

            a = pos_proj[0, 0]
            b = pos_proj[1, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[0, 0] = b / radius
            rotation[0, 1] = -a / radius
            rotation[1, 0] = a / radius
            rotation[1, 1] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

            a = pos_proj[1, 0]
            b = pos_proj[2, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[1, 1] = b / radius
            rotation[1, 2] = -a / radius
            rotation[2, 1] = a / radius
            rotation[2, 2] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

            a = pos_proj[2, 0]
            b = pos_proj[3, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[2, 2] = b / radius
            rotation[2, 3] = -a / radius
            rotation[3, 2] = a / radius
            rotation[3, 3] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

            # Rows zero and one already have zero first coordinates, so their
            # remaining rotation can eliminate another projection scalar while
            # preserving all three first-column constraints.
            a = pos_proj[0, 1]
            b = pos_proj[1, 1]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[0, 0] = b / radius
            rotation[0, 1] = -a / radius
            rotation[1, 0] = a / radius
            rotation[1, 1] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

            # Rows one and two remain zero in the first column, so rotate them
            # to eliminate another second-column scalar without disturbing any
            # of the four previously fixed entries.
            a = pos_proj[1, 1]
            b = pos_proj[2, 1]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[1, 1] = b / radius
            rotation[1, 2] = -a / radius
            rotation[2, 1] = a / radius
            rotation[2, 2] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

            # Rows zero and one are both zero in the first two columns, leaving
            # one residual rotation that can eliminate a third-column scalar.
            a = pos_proj[0, 2]
            b = pos_proj[1, 2]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(4, device=pos_proj.device, dtype=pos_proj.dtype)
            rotation[0, 0] = b / radius
            rotation[0, 1] = -a / radius
            rotation[1, 0] = a / radius
            rotation[1, 1] = b / radius
            pos_code = pos_code @ rotation.T
            pos_proj = rotation @ pos_proj

        # Generic fixed harmonics expose both position sums and differences to
        # the learned query/key bilinear forms without a learned code per slot.
        positions = torch.arange(
            cfg.max_seq_len,
            device=self.pos_emb.weight.device,
            dtype=self.pos_emb.weight.dtype,
        )
        phase = 2.0 * math.pi * positions / cfg.max_seq_len
        pos_code = torch.stack(
            (
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(2.0 * phase),
                torch.cos(2.0 * phase),
            ),
            dim=-1,
        ) / math.sqrt(2.0)
        self.register_buffer("pos_code", pos_code)
        # A position-dependent shift shared by all model coordinates is removed
        # by every downstream LayerNorm. Center each readout row to fix this
        # exact gauge, then omit its final coordinate.
        pos_proj = self.pos_emb.weight[:4].detach().clone()
        pos_proj = pos_proj - pos_proj.mean(dim=1, keepdim=True)
        self.pos_proj = nn.Parameter(pos_proj[:, :-1].clone())
        self.pos_emb = None

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
        pos_proj = torch.cat(
            (self.pos_proj, -self.pos_proj.sum(dim=1, keepdim=True)), dim=1
        )
        position = F.embedding(pos, self.pos_code) @ pos_proj
        x = self.token_emb(idx) + position
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
