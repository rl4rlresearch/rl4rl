"""
Decoder-only transformer for 10-digit addition.

Architecture: GPT-style with pre-LayerNorm, causal self-attention,
learned positional embeddings, and weight tying (embedding = output head).
All linear layers use bias=False to minimize parameter count.

The final model: d=16, h=2, L=2, ff=48 = 6,080 parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedSeparatorEmbedding(nn.Module):
    """Learned digit rows with both input-only separators fixed."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.learned_weight = nn.Parameter(torch.empty(10, embedding_dim))
        self.register_buffer('plus_weight', torch.zeros(1, embedding_dim))
        self.register_buffer('equals_weight', torch.zeros(1, embedding_dim))

    def forward(self, indices):
        weight = torch.cat(
            (self.learned_weight, self.plus_weight, self.equals_weight), dim=0)
        return F.embedding(indices, weight)


class TiedLearnedHead(nn.Module):
    """Output over the learned rows; '=' is input-only."""

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, x):
        return F.linear(x, self.embedding.learned_weight)


class StructuredPositionEmbedding(nn.Module):
    """Position table with tied rows and deterministic interpolated rows."""

    def __init__(self, num_embeddings, embedding_dim, tied_pairs,
                 interpolated_positions):
        super().__init__()
        pairs = [tuple(sorted(pair)) for pair in tied_pairs]
        assert all(0 <= low < high < num_embeddings for low, high in pairs)
        removed = {high for _, high in pairs}
        assert len(removed) == len(pairs)
        interpolated = set(interpolated_positions)
        assert not (removed & interpolated)
        kept = [position for position in range(num_embeddings)
                if position not in removed and position not in interpolated]
        target_index = {position: index for index, position in enumerate(kept)}
        self.weight = nn.Parameter(
            torch.empty(len(kept), embedding_dim))
        representative = {high: low for low, high in pairs}
        basis = torch.zeros(num_embeddings, len(kept))
        for position in range(num_embeddings):
            if position in representative:
                basis[position, target_index[representative[position]]] = 1.0
            elif position in interpolated:
                left = max(anchor for anchor in kept if anchor < position)
                right = min(anchor for anchor in kept if anchor > position)
                span = right - left
                basis[position, target_index[left]] = (right - position) / span
                basis[position, target_index[right]] = (position - left) / span
            else:
                basis[position, target_index[position]] = 1.0
        self.register_buffer('basis', basis)
        self.register_buffer('anchor_positions', torch.tensor(kept))

    def forward(self, indices):
        return self.basis[indices] @ self.weight


class FactorizedLinear(nn.Module):
    """Bias-free low-rank linear map."""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.first = nn.Linear(in_features, rank, bias=False)
        self.second = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.second(self.first(x))


class ZeroAttention(nn.Module):
    """Parameter-free residual branch used for a first-layer ablation."""

    def forward(self, x):
        return torch.zeros_like(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, max_seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_dim, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size=15, d_model=128, n_heads=4, n_layers=4,
                 ff_dim=512, max_seq_len=40, dropout=0.0,
                 block1_ff_dim=None, fixed_separators=False,
                 tied_position_pairs=None, interpolated_positions=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = (
            FixedSeparatorEmbedding(d_model)
            if fixed_separators
            else nn.Embedding(vocab_size, d_model)
        )
        self.pos_emb = (
            StructuredPositionEmbedding(
                max_seq_len, d_model, tied_position_pairs,
                interpolated_positions or [])
            if tied_position_pairs is not None
            else nn.Embedding(max_seq_len, d_model)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        if block1_ff_dim is not None:
            self.blocks[1].ff = nn.Sequential(
                nn.Linear(d_model, block1_ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(block1_ff_dim, d_model, bias=False),
            )
        self.blocks[0].attn = ZeroAttention()
        self.blocks[0].ln1 = nn.Identity()
        self.blocks[1].attn.proj = FactorizedLinear(
            d_model, d_model, rank=7)

        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False)
        if fixed_separators:
            self.head = TiedLearnedHead(self.token_emb)
        else:
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, FixedSeparatorEmbedding):
            torch.nn.init.normal_(module.learned_weight, mean=0.0, std=0.02)
        elif isinstance(module, StructuredPositionEmbedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id=None):
        """Autoregressive generation (greedy argmax)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
