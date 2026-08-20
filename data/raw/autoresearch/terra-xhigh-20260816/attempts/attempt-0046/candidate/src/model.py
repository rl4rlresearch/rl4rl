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
    def __init__(self, d_model, n_heads, ff_dim, max_seq_len, dropout=0.0,
                 no_norm_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, bias=not no_norm_bias)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model, bias=not no_norm_bias)
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
                 ff_dim=512, max_seq_len=40, dropout=0.0, position_rank=None,
                 tie_norms_across_layers=False, no_norm_bias=False,
                 factorized_positions=False, tie_operand_segments=False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.position_rank = position_rank
        self.factorized_positions = factorized_positions
        self.tie_operand_segments = tie_operand_segments
        if position_rank is None:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.pos_index_emb = None
            self.pos_segment_emb = None
            self.pos_proj = None
        else:
            if factorized_positions:
                self.pos_emb = None
                self.pos_index_emb = nn.Embedding(11, position_rank)
                segment_count = 6 if tie_operand_segments else 7
                self.pos_segment_emb = nn.Embedding(segment_count, position_rank)
                self.register_buffer(
                    'position_index',
                    torch.tensor(
                        [0] + list(range(10)) + [0] + list(range(10))
                        + [0] + list(range(11)) + [0],
                        dtype=torch.long,
                    ),
                )
                if tie_operand_segments:
                    position_segment = (
                        [0] + [1] * 10 + [2] + [1] * 10 + [3]
                        + [4] * 11 + [5]
                    )
                else:
                    position_segment = (
                        [0] + [1] * 10 + [2] + [3] * 10 + [4]
                        + [5] * 11 + [6]
                    )
                self.register_buffer(
                    'position_segment', torch.tensor(position_segment, dtype=torch.long)
                )
            else:
                self.pos_emb = nn.Embedding(max_seq_len, position_rank)
                self.pos_index_emb = None
                self.pos_segment_emb = None
            self.pos_proj = nn.Linear(position_rank, d_model, bias=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, ff_dim, max_seq_len, dropout, no_norm_bias
            )
            for _ in range(n_layers)
        ])
        if tie_norms_across_layers:
            assert n_layers >= 2
            self.blocks[1].ln1 = self.blocks[0].ln1
            self.blocks[1].ln2 = self.blocks[0].ln2

        self.ln_f = nn.LayerNorm(d_model, bias=not no_norm_bias)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying between token embedding and output head
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        if self.factorized_positions:
            pos_emb = (
                self.pos_index_emb(self.position_index[:T])
                + self.pos_segment_emb(self.position_segment[:T])
            )
        else:
            pos_emb = self.pos_emb(pos)
        if self.pos_proj is not None:
            pos_emb = self.pos_proj(pos_emb)
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
