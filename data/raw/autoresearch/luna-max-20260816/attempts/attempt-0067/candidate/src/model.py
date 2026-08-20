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
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0,
                 shared_output_head=None, shared_output_head_1=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * self.d_head, bias=False)
        self.grouped_proj = nn.ModuleList([
            shared_output_head if head == 0 and shared_output_head is not None
            else shared_output_head_1 if head == 1 and shared_output_head_1 is not None
            else nn.Linear(self.d_head, self.d_head, bias=False)
            for head in range(n_heads)
        ])
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.d_head)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, T, 2, 1, self.d_head)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        if self.n_heads > 1:
            k = k.expand(B, self.n_heads, T, self.d_head)
            v = v.expand(B, self.n_heads, T, self.d_head)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = torch.stack(
            [layer(y[:, head]) for head, layer in enumerate(self.grouped_proj)],
            dim=1,
        )
        return y.transpose(1, 2).contiguous().view(B, T, C)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, max_seq_len, dropout=0.0,
                 shared_tied_weight=None, shared_output_head=None,
                 shared_output_head_1=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout,
            shared_output_head=shared_output_head,
            shared_output_head_1=shared_output_head_1,
        )
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        if shared_tied_weight is None:
            self.ff_tied = nn.Parameter(torch.empty(d_model))
        else:
            self.ff_tied = shared_tied_weight
        self.ff_in = nn.Linear(d_model, ff_dim - 1, bias=False)
        self.ff_out = nn.Linear(ff_dim - 1, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        normed = self.ln2(x)
        tied_hidden = F.gelu(F.linear(normed, self.ff_tied.unsqueeze(0)))
        free = self.ff_out(F.gelu(self.ff_in(normed)))
        tied = F.linear(tied_hidden, self.ff_tied.unsqueeze(1))
        x = x + self.dropout(tied + free)
        return x


class AdditionTransformer(nn.Module):
    def __init__(self, vocab_size=15, d_model=128, n_heads=4, n_layers=4,
                 ff_dim=512, max_seq_len=40, dropout=0.0, pos_rank=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pos_rank = pos_rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        if pos_rank is None:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.pos_up = None
        else:
            self.pos_emb = nn.Embedding(max_seq_len, pos_rank)
            self.pos_up = nn.Linear(pos_rank, d_model, bias=False)

        shared_tied_weight = nn.Parameter(torch.empty(d_model))
        torch.nn.init.normal_(shared_tied_weight, mean=0.0, std=0.02)
        self.shared_tied_weight = shared_tied_weight
        shared_output_head = nn.Linear(d_model // n_heads, d_model // n_heads,
                                       bias=False)
        self.shared_output_head = shared_output_head
        shared_output_head_1 = nn.Linear(d_model // n_heads, d_model // n_heads,
                                         bias=False)
        self.shared_output_head_1 = shared_output_head_1
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, max_seq_len, dropout,
                             shared_tied_weight=shared_tied_weight,
                             shared_output_head=shared_output_head,
                             shared_output_head_1=shared_output_head_1)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln_f_scale = nn.Parameter(torch.ones(n_heads))
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
        if self.pos_up is not None:
            pos_emb = self.pos_up(pos_emb)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        x = x * self.ln_f_scale.repeat_interleave(self.d_model // self.blocks[0].attn.n_heads)
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
