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


class SparseBiasLayerNorm(nn.Module):
    """LayerNorm with selected bias coordinates fixed exactly to zero."""
    def __init__(self, d_model, zero_indices=(), fixed_weight_indices=(), eps=1e-5):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        fixed_weight_indices = set(fixed_weight_indices)
        if fixed_weight_indices:
            weight_keep = [i for i in range(d_model) if i not in fixed_weight_indices]
            self.weight_values = nn.Parameter(torch.ones(len(weight_keep)))
            self.register_buffer(
                'weight_indices', torch.tensor(weight_keep, dtype=torch.long))
            self.weight = None
        else:
            self.weight = nn.Parameter(torch.ones(d_model))
        keep = [i for i in range(d_model) if i not in set(zero_indices)]
        self.bias_values = nn.Parameter(torch.zeros(len(keep)))
        self.register_buffer('bias_indices', torch.tensor(keep, dtype=torch.long))

    def forward(self, x):
        if self.weight is None:
            weight = self.weight_values.new_ones(self.d_model)
            weight = weight.scatter(0, self.weight_indices, self.weight_values)
        else:
            weight = self.weight
        bias = weight.new_zeros(weight.shape)
        bias = bias.scatter(0, self.bias_indices, self.bias_values)
        return F.layer_norm(x, weight.shape, weight, bias, self.eps)


class TiedPositionEmbedding(nn.Module):
    """Embedding table reconstructed from honestly counted shared scalars."""
    def __init__(self, num_embeddings, embedding_dim, tied_pairs=()):
        super().__init__()
        size = num_embeddings * embedding_dim
        parent = list(range(size))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left, right in tied_pairs:
            parent[find(right)] = find(left)
        group_ids = {}
        mapping = []
        for index in range(size):
            root = find(index)
            if root not in group_ids:
                group_ids[root] = len(group_ids)
            mapping.append(group_ids[root])
        self.values = nn.Parameter(torch.zeros(len(group_ids)))
        self.register_buffer(
            'value_indices', torch.tensor(mapping, dtype=torch.long).view(
                num_embeddings, embedding_dim))

    def forward(self, indices):
        weight = self.values[self.value_indices]
        return F.embedding(indices, weight)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Identity()
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
                 block_index=0):
        super().__init__()
        self.ln1 = SparseBiasLayerNorm(
            d_model,
            zero_indices=((7, 8) if block_index == 0 else (4, 5, 6, 10)),
            fixed_weight_indices=(() if block_index == 0 else (11,)),
        )
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = (SparseBiasLayerNorm(
                        d_model, zero_indices=(9, 11), fixed_weight_indices=(12,))
                    if block_index == 1 else nn.LayerNorm(d_model))
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
                 ff_dim=512, max_seq_len=40, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = TiedPositionEmbedding(
            max_seq_len, d_model,
            tied_pairs=((77, 498), (183, 282), (181, 302), (49, 125),
                        (388, 250), (451, 367), (450, 418), (237, 222),
                        (70, 183), (409, 208), (500, 273), (250, 38),
                        (335, 48),
                        (364, 501),
                        (177, 462),
                        (211, 348),
                        (260, 244),
                        (188, 21),
                        (238, 274),
                        (194, 177),
                        (334, 403),
                        (175, 208),
                        (442, 179),
                        (41, 106),
                        (95, 424),
                        (275, 10),
                        (471, 445),
                        (24, 113),
                        (294, 314),
                        (397, 506),
                        (176, 134),
                        (379, 223),
                        (7, 465),
                        (66, 312),
                        (64, 330),
                        (233, 306),
                        (361, 469),
                        (473, 99),
                        (25, 454),
                        (142, 169),
                        (441, 173),
                        (139, 279),
                        (391, 353),
                        (452, 447),
                        (217, 482),
                        (386, 83),
                        (33, 429),
                        (105, 290),
                        (50, 161),
                        (509, 84),
                        (211, 41),
                        (271, 254),
                        (418, 57),
                        (131, 119),
                        (110, 150),
                        (406, 94),
                        (253, 151),
                        (286, 159),
                        (190, 115),
                        (64, 378),
                        (417, 13),
                        (304, 439),
                        (66, 504),
                        (301, 7),
                        (65, 399),
                        (266, 375),
                        (104, 175),
                        (298, 494),
                        (280, 72),
                        (447, 37),
                        (62, 247),
                        (478, 264),
                        (436, 16),
                        (152, 366),
                        (154, 288),
                        (248, 118),
                        (192, 63),
                        (7, 166),
                        (197, 398),
                        (112, 128),
                        (56, 126),
                        (195, 179),
                        (317, 336),
                        (376, 90),
                        (244, 488),
                        (177, 41),
                        (383, 265),
                        (323, 381),
                        (320, 257),
                        (287, 322),
                        (77, 291),
                        (434, 430),
                        (88, 25),
                        (433, 266),
                        (397, 246),
                        (120, 255),
                        (34, 304),
                        (463, 138),
                        (401, 263),
                        (367, 435),
                        (198, 273),
                        (66, 269),
                        (382, 182),
                        (0, 421),
                        (136, 285),
                        (210, 256),
                        (295, 422),
                        (58, 97),
                        (262, 289),
                        (144, 79),
                        (393, 122),
                        (319, 7),
                        (145, 142),
                        (407, 109),
                        (110, 112),
                        (118, 474),
                        (224, 207),
                        (359, 467),
                        (321, 363),
                        (327, 119),
                        (63, 134),
                        (168, 477),
                        (428, 264),
                        (186, 18),
                        (459, 497),
                        (464, 31),
                        (346, 254),
                        (318, 405),
                        (114, 137),
                        (231, 13),
                        (90, 141),
                        (449, 54),
                        (40, 485),
                        (154, 61),
                        (83, 67),
                        (57, 456),
                        (120, 157),
                        (215, 202),
                        (440, 328),
                        (16, 491),
                        (25, 105),
                        (394, 41),
                        (377, 66),
                        (60, 6),
                        (170, 392),
                        (344, 136),
                        (445, 455),
                        (77, 362),
                        (156, 343),
                        (315, 331),
                        (9, 95),
                        (151, 410),
                        (487, 334),
                        (222, 50),
                        (96, 297),
                        (483, 214),
                        (283, 373),
                        (79, 1),
                        (246, 70),
                        (68, 384),
                        (251, 480),
                        (51, 205),
                        (104, 197),
                        (32, 220),
                        (138, 127),
                        (160, 438),
                        (360, 430),
                        (181, 207),
                        (415, 227),
                        (111, 294),
                        (22, 257),
                        (298, 367),
                        (326, 223),
                        (295, 423),
                        (341, 121),
                        (243, 272),
                        (64, 94),
                        (374, 15),
                        (173, 37),
                        (300, 16),
                        (380, 158),
                        (359, 329),
                        (287, 217),
                        (361, 179),
                        (56, 82),
                        (296, 458),
                        (402, 153),
                        (163, 24),
                        (40, 426),
                        (216, 266),
                        (385, 353),
                        (110, 369),
                        (233, 303),
                        (204, 72),
                        (48, 259),
                        (99, 67),
                        (120, 143),
                        (45, 108),
                        (142, 495),
                        (111, 10),
                        (146, 122),
                        (90, 191),
                        (152, 365),
                        (349, 114),
                        (419, 116),
                        (51, 81),
                        (218, 321),
                        (86, 199),
                        (408, 315),
                        (227, 210),
                        (100, 52),
                        (481, 18),
                        (49, 254),
                        (490, 182),
                        (193, 102),
                        (216, 459),
                        (43, 69),
                        (181, 239),
                        (168, 284),
                        (80, 278),
                        (184, 475),
                        (178, 118),
                        (240, 120),
                        (351, 236),
                        (263, 298),
                        (47, 507),
                        (238, 457),
                        (13, 307),
                        (425, 104),
                        (139, 11),
                        (201, 431),
                        (295, 262),
                        (189, 258),
                        (470, 486),
                        (489, 158),
                        (225, 241),
                        (160, 233),
                        (445, 387),
                        (7, 492),
                        (54, 78),
                        (87, 71),
                        (74, 223),
                        (61, 9),
                        (446, 8),
                        (147, 206),
                        (5, 15),
                        (151, 162),
                        (38, 62),
                        (466, 57),
                        (234, 413)))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, max_seq_len, dropout, i)
            for i in range(n_layers)
        ])

        # Reuse the learned query/key/value transform across depth. Residual
        # streams, normalizers, attention maps, and feed-forward paths remain
        # layer-specific, so the two blocks can still perform different work.
        for block in self.blocks[1:]:
            block.attn.qkv.weight = self.blocks[0].attn.qkv.weight
            block.ff[0].weight = self.blocks[0].ff[0].weight
            block.ff[2].weight = self.blocks[0].ff[2].weight

        self.ln_f = nn.LayerNorm(d_model, bias=False)
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
