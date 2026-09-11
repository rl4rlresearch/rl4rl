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
        # full direct query bias and independent query/key projections while
        # sharing one zero-mean value readout across the routing heads.
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

        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next five
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 19))
        self.relative_bias_core_eighteenth = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_preantepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_antepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))

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
            # Combine the initialized per-head value maps at variance-preserving
            # scale, then learn a single semantic readout used by both routes.
            v_weight = v_weight.view(
                self.n_head, self.head_dim, d_model
            ).sum(dim=0) / math.sqrt(self.n_head)
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
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight, self.qkv.bias)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias_core_eighteenth = (
            self.relative_bias_core_eighteenth.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_preantepenultimate = (
            self.relative_bias_core_preantepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_antepenultimate = (
            self.relative_bias_core_antepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_penultimate = (
            self.relative_bias_core_penultimate.expand(self.n_head).unsqueeze(1)
        )
        relative_bias_core_endpoint = self.relative_bias_core_endpoint.expand(
            self.n_head
        ).unsqueeze(1)
        relative_bias_penultimate = torch.cat(
            (
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
                self.relative_bias_endpoint,
            )
        ).unsqueeze(1)
        relative_bias_endpoint = torch.cat(
            (
                self.relative_bias_endpoint,
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
            )
        ).unsqueeze(1)
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_eighteenth,
                relative_bias_core_preantepenultimate,
                relative_bias_core_antepenultimate,
                relative_bias_core_penultimate,
                relative_bias_core_endpoint,
                relative_bias_penultimate,
                relative_bias_endpoint,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
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

        # The downstream LayerNorm removes the all-ones component of this
        # bias. Retain only its seven observable zero-mean coordinates.
        self.fc2.bias = nn.Parameter(self.fc2.bias[:-1].clone())

    def gauge_fix_fc1(self) -> None:
        with torch.no_grad():
            weight = self.fc1.weight
            centered = weight - weight.mean(dim=1, keepdim=True)
            self.fc1.weight = nn.Parameter(
                (centered @ self.fc1_basis.T).clone()
            )

    def gauge_fix_fc2(self) -> None:
        with torch.no_grad():
            weight = self.fc2.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.fc2.weight = nn.Parameter(
                (self.fc1_basis @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), fc2_weight, fc2_bias)
        return self.drop(output)


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

        # The final state reaches the logits only through a rank-six lexical
        # projection. Learn its observable bias directly in that space instead
        # of retaining two null directions in the LayerNorm bias.
        token_rank = 6
        self.ln_f_token_bias = nn.Parameter(
            self.ln_f.bias.new_zeros(token_rank)
        )
        self.ln_f.bias = None

        # Initialize a dense tied lexical table as before, then compress it
        # below into a shared learned latent representation.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Compress only after full initialization, preserving the original RNG
        # stream and the initialized function on LayerNorm outputs.
        for block in self.blocks:
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
            block.mlp.gauge_fix_fc2()

        # A rank-six product is invariant under an invertible change of latent
        # basis. Fix that 36-dimensional gauge by choosing the best-conditioned
        # six feature columns as an identity pivot and learning only the two
        # remaining columns.
        with torch.no_grad():
            token_left, token_singular, token_right = torch.linalg.svd(
                self.token_emb.weight, full_matrices=False
            )
            token_scale = token_singular[:token_rank].sqrt()
            token_code = token_left[:, :token_rank] * token_scale
            token_proj = token_scale.unsqueeze(1) * token_right[:token_rank]

            candidates = torch.combinations(
                torch.arange(cfg.d_model, device=token_proj.device),
                r=token_rank,
            )
            candidate_matrices = token_proj[:, candidates].permute(1, 0, 2)
            pivot_columns = candidates[
                torch.linalg.det(candidate_matrices).abs().argmax()
            ]
            all_columns = torch.arange(cfg.d_model, device=token_proj.device)
            tail_columns = all_columns[
                (all_columns[:, None] != pivot_columns[None, :]).all(dim=1)
            ]
            column_order = torch.cat((pivot_columns, tail_columns))
            pivot = token_proj[:, pivot_columns]

            token_code = token_code @ pivot
            token_tail = torch.linalg.solve(
                pivot, token_proj[:, tail_columns]
            )

        self.token_code = nn.Parameter(token_code.clone())
        self.token_proj = nn.Parameter(token_tail.clone())
        self.register_buffer(
            "token_pivot",
            torch.eye(
                token_rank,
                device=token_proj.device,
                dtype=token_proj.dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "token_inverse_order", torch.argsort(column_order)
        )
        self.token_emb = None
        self.lm_head = None

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

        # Give each harmonic a fixed orthogonal direction in the observable
        # zero-mean residual subspace. Learned query/key maps can perform the
        # required mixing, so retain only one trainable amplitude per harmonic.
        pos_basis = torch.eye(
            cfg.d_model,
            device=self.pos_emb.weight.device,
            dtype=self.pos_emb.weight.dtype,
        )[:, :4]
        pos_basis = pos_basis - pos_basis.mean(dim=0, keepdim=True)
        pos_basis, _ = torch.linalg.qr(pos_basis, mode="reduced")
        self.register_buffer(
            "pos_basis", pos_basis.T.contiguous(), persistent=False
        )

        initialized_pos_rows = self.pos_emb.weight[:4].detach().clone()
        initialized_pos_rows = initialized_pos_rows - initialized_pos_rows.mean(
            dim=1, keepdim=True
        )
        self.pos_scale = nn.Parameter(
            torch.linalg.vector_norm(initialized_pos_rows, dim=1)
        )
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
        pos_proj = self.pos_scale.unsqueeze(1) * self.pos_basis
        position = F.embedding(pos, self.pos_code) @ pos_proj
        token_proj = torch.cat(
            (self.token_pivot, self.token_proj), dim=1
        )[:, self.token_inverse_order]
        token = F.embedding(idx, self.token_code) @ token_proj
        x = token + position
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        token_features = x @ token_proj.T + self.ln_f_token_bias
        logits = F.linear(token_features, self.token_code)

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
