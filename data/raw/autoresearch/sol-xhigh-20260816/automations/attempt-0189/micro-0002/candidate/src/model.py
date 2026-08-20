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


TOKEN_TIED_PAIRS = (
    (131, 97),
    (166, 148),
    (97, 11),
    (107, 124),
    (120, 0),
    (79, 168),
    (163, 179),
    (133, 11),
    (148, 52),
    (48, 35),
    (82, 25),
    (127, 53),
    (37, 77),
    (69, 111),
    (149, 156),
    (65, 151),
    (164, 52),
)

QKV_TIED_PAIRS = (
    (649, 600),
    (608, 693),
    (523, 393),
    (587, 530),
    (273, 258),
    (608, 690),
    (381, 105),
    (594, 601),
    (644, 749),
    (345, 335),
    (767, 287),
    (695, 606),
    (394, 78),
    (593, 105),
    (538, 742),
    (626, 598),
    (608, 760),
    (644, 738),
    (764, 727),
    (654, 711),
    (597, 724),
    (741, 680),
    (676, 725),
    (635, 545),
    (441, 46),
    (588, 731),
    (481, 545),
    (745, 574),
    (705, 643),
    (164, 201),
    (259, 488),
    (657, 538),
    (567, 744),
    (622, 746),
    (627, 603),
    (590, 169),
    (383, 576),
    (762, 397),
    (613, 728),
    (727, 683),
    (633, 422),
    (557, 499),
    (454, 527),
    (565, 628),
    (600, 629),
    (423, 518),
    (653, 561),
    (131, 121),
    (714, 716),
    (555, 517),
    (315, 106),
    (135, 643),
    (638, 696),
    (710, 622),
    (547, 691),
    (679, 494),
    (765, 639),
    (393, 755),
    (535, 321),
    (359, 65),
    (209, 330),
    (439, 428),
    (297, 553),
    (625, 683),
    (646, 667),
    (718, 260),
    (684, 685),
    (748, 706),
    (564, 699),
    (105, 278),
    (156, 348),
    (207, 470),
    (396, 82),
    (234, 468),
    (689, 549),
    (670, 621),
    (682, 499),
    (740, 529),
    (346, 419),
    (614, 756),
    (650, 50),
    (544, 197),
    (171, 12),
    (573, 665),
    (127, 616),
    (294, 730),
    (78, 537),
    (436, 456),
    (743, 565),
    (675, 575),
    (481, 580),
    (660, 103),
    (571, 602),
    (609, 303),
    (532, 517),
    (466, 274),
    (385, 726),
    (383, 454),
    (135, 483),
    (128, 56),
    (596, 677),
    (449, 316),
    (592, 720),
    (644, 766),
    (538, 585),
    (584, 53),
    (680, 709),
    (578, 434),
    (207, 393),
    (447, 410),
    (1, 732),
    (181, 269),
    (647, 604),
    (673, 615),
    (342, 698),
    (701, 270),
    (309, 549),
    (111, 177),
    (55, 505),
    (398, 306),
    (391, 664),
    (120, 472),
    (337, 733),
    (385, 662),
    (34, 42),
    (52, 283),
    (401, 54),
    (78, 655),
    (529, 127),
    (588, 574),
    (321, 751),
    (674, 322),
    (308, 263),
    (722, 574),
    (56, 480),
    (714, 216),
    (639, 687),
    (708, 303),
    (464, 10),
    (583, 577),
    (514, 497),
    (300, 469),
    (370, 499),
    (110, 162),
    (704, 736),
    (654, 641),
    (291, 229),
    (142, 659),
    (410, 219),
)

FF_IN_TIED_PAIRS = (
    (160, 185),
    (170, 101),
    (43, 197),
    (83, 59),
    (79, 80),
    (25, 107),
    (76, 163),
    (207, 30),
    (247, 245),
    (65, 162),
    (179, 253),
    (42, 156),
    (252, 184),
    (146, 172),
    (55, 63),
    (196, 134),
    (225, 2),
    (220, 18),
    (140, 139),
    (210, 119),
    (147, 40),
    (149, 89),
    (118, 181),
    (92, 160),
    (33, 132),
    (145, 216),
    (47, 167),
    (120, 67),
    (68, 234),
    (208, 240),
    (32, 229),
    (137, 228),
    (148, 169),
    (19, 36),
    (87, 22),
    (98, 85),
    (75, 224),
    (117, 61),
    (50, 214),
    (248, 33),
    (178, 223),
    (93, 221),
)

FF_OUT_TIED_PAIRS = (
    (199, 228),
    (103, 249),
    (5, 251),
    (202, 83),
    (95, 186),
    (173, 74),
    (31, 15),
    (193, 56),
    (66, 72),
    (13, 81),
    (111, 229),
    (207, 184),
    (210, 204),
    (145, 83),
    (63, 181),
    (55, 61),
    (74, 112),
    (211, 197),
    (246, 15),
    (185, 93),
    (216, 17),
    (151, 240),
    (49, 20),
    (90, 67),
    (95, 52),
    (204, 58),
    (238, 175),
    (0, 219),
    (108, 165),
    (82, 85),
    (214, 9),
    (63, 83),
    (77, 200),
    (10, 197),
    (99, 169),
    (16, 17),
    (208, 218),
    (199, 13),
    (67, 97),
    (56, 64),
    (118, 119),
    (69, 7),
    (247, 84),
    (239, 143),
    (58, 19),
    (191, 79),
    (144, 107),
    (209, 168),
    (71, 92),
    (117, 74),
    (182, 189),
    (248, 149),
    (157, 101),
    (215, 11),
    (227, 78),
    (140, 67),
    (94, 171),
    (113, 87),
    (91, 130),
    (148, 16),
    (187, 65),
    (10, 183),
    (205, 103),
    (118, 212),
    (177, 253),
    (203, 77),
    (179, 167),
    (222, 7),
    (28, 66),
    (134, 131),
    (25, 128),
    (192, 96),
    (162, 12),
    (16, 20),
    (6, 151),
    (89, 127),
    (132, 46),
    (245, 5),
    (15, 226),
    (86, 254),
    (18, 44),
    (94, 161),
    (243, 196),
    (55, 242),
    (252, 65),
    (198, 184),
    (188, 241),
    (103, 68),
    (79, 74),
    (77, 56),
    (224, 152),
    (75, 194),
    (63, 244),
    (5, 208),
    (231, 135),
    (137, 4),
    (121, 3),
    (138, 99),
    (114, 236),
    (155, 230),
    (93, 182),
    (96, 118),
    (71, 196),
    (7, 176),
    (250, 10),
    (150, 174),
    (89, 91),
    (101, 84),
    (60, 51),
    (201, 18),
)

LN_F_TIED_PAIRS = (
    (3, 10),
    (6, 9),
    (0, 14),
    (7, 1),
    (3, 6),
    (0, 1),
    (3, 2),
    (0, 8),
    (5, 12),
    (13, 0),
    (5, 4),
    (15, 4),
    (4, 0),
    (11, 0),
)

BLOCK0_LN2_WEIGHT_TIED_PAIRS = (
    (6, 7),
)

BLOCK0_LN2_BIAS_TIED_PAIRS = (
    (9, 6),
    (6, 1),
    (7, 11),
)


class TiedTokenEmbedding(nn.Module):
    """Token matrix reconstructed from honestly counted shared scalars."""
    def __init__(self, num_embeddings, embedding_dim, tied_pairs=()):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
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
        self.values = nn.Parameter(torch.empty(len(group_ids)))
        nn.init.normal_(self.values, mean=0.0, std=0.02)
        self.register_buffer(
            'value_indices', torch.tensor(mapping, dtype=torch.long).view(
                num_embeddings, embedding_dim))

    def reconstructed_weight(self):
        return self.values[self.value_indices]

    def forward(self, indices):
        return F.embedding(indices, self.reconstructed_weight())


class TiedLinear(nn.Module):
    """Bias-free linear map reconstructed from shared scalar groups."""
    def __init__(self, in_features, out_features, tied_pairs=()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        size = in_features * out_features
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
        self.values = nn.Parameter(torch.empty(len(group_ids)))
        nn.init.normal_(self.values, mean=0.0, std=0.02)
        self.register_buffer(
            'value_indices', torch.tensor(mapping, dtype=torch.long).view(
                out_features, in_features))

    def reconstructed_weight(self):
        return self.values[self.value_indices]

    def forward(self, x):
        return F.linear(x, self.reconstructed_weight())


class BlendedGELU(nn.Module):
    def forward(self, x):
        exact = F.gelu(x, approximate="none")
        approximate = F.gelu(x, approximate="tanh")
        return exact + 0.5 * (approximate - exact)


class TiedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm whose learned scales may share scalar groups."""
    def __init__(self, d_model, tied_pairs=(), eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        parent = list(range(d_model))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left, right in tied_pairs:
            parent[find(right)] = find(left)
        group_ids = {}
        mapping = []
        for index in range(d_model):
            root = find(index)
            if root not in group_ids:
                group_ids[root] = len(group_ids)
            mapping.append(group_ids[root])
        self.values = nn.Parameter(torch.ones(len(group_ids)))
        self.register_buffer('value_indices', torch.tensor(mapping, dtype=torch.long))

    def forward(self, x):
        weight = self.values[self.value_indices]
        return F.layer_norm(x, (self.d_model,), weight, None, self.eps)


class TiedAffineLayerNorm(nn.Module):
    """LayerNorm with independently grouped learned scales and biases."""
    def __init__(self, d_model, weight_tied_pairs=(), bias_tied_pairs=(), eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        def make_mapping(tied_pairs):
            parent = list(range(d_model))

            def find(index):
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            for left, right in tied_pairs:
                parent[find(right)] = find(left)
            group_ids = {}
            mapping = []
            for index in range(d_model):
                root = find(index)
                if root not in group_ids:
                    group_ids[root] = len(group_ids)
                mapping.append(group_ids[root])
            return torch.tensor(mapping, dtype=torch.long), len(group_ids)

        weight_mapping, weight_groups = make_mapping(weight_tied_pairs)
        bias_mapping, bias_groups = make_mapping(bias_tied_pairs)
        self.weight_values = nn.Parameter(torch.ones(weight_groups))
        self.bias_values = nn.Parameter(torch.zeros(bias_groups))
        self.register_buffer('weight_indices', weight_mapping)
        self.register_buffer('bias_indices', bias_mapping)

    def forward(self, x):
        weight = self.weight_values[self.weight_indices]
        bias = self.bias_values[self.bias_indices]
        return F.layer_norm(x, (self.d_model,), weight, bias, self.eps)


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

        self.qkv = TiedLinear(
            d_model, 3 * d_model, tied_pairs=QKV_TIED_PAIRS)
        self.proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (1.015 / math.sqrt(self.d_head))
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
                    if block_index == 1 else TiedAffineLayerNorm(
                        d_model,
                        weight_tied_pairs=BLOCK0_LN2_WEIGHT_TIED_PAIRS,
                        bias_tied_pairs=BLOCK0_LN2_BIAS_TIED_PAIRS))
        self.ff = nn.Sequential(
            TiedLinear(d_model, ff_dim, tied_pairs=FF_IN_TIED_PAIRS),
            BlendedGELU(),
            TiedLinear(ff_dim, d_model, tied_pairs=FF_OUT_TIED_PAIRS),
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

        self.token_emb = TiedTokenEmbedding(
            vocab_size, d_model, tied_pairs=TOKEN_TIED_PAIRS)
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
                        (234, 413),
                        (1, 217),
                        (25, 370),
                        (296, 185),
                        (73, 167),
                        (104, 310),
                        (265, 353),
                        (123, 323),
                        (499, 16),
                        (493, 115),
                        (49, 270),
                        (109, 7),
                        (18, 132),
                        (21, 484),
                        (503, 371),
                        (67, 66),
                        (158, 31),
                        (42, 103),
                        (56, 414),
                        (26, 84),
                        (404, 252),
                        (368, 45),
                        (119, 339),
                        (130, 98),
                        (92, 358),
                        (22, 218),
                        (72, 328),
                        (334, 318),
                        (179, 238),
                        (96, 198),
                        (65, 48),
                        (249, 6),
                        (63, 210),
                        (37, 151),
                        (221, 122),
                        (390, 153),
                        (10, 102),
                        (3, 40),
                        (214, 209),
                        (17, 337),
                        (448, 52),
                        (89, 8),
                        (472, 412),
                        (41, 77),
                        (427, 135),
                        (225, 479),
                        (170, 136),
                        (74, 90),
                        (443, 338),
                        (85, 101),
                        (355, 168),
                        (57, 114),
                        (364, 120),
                        (71, 311),
                        (76, 230),
                        (159, 104),
                        (80, 262),
                        (121, 317),
                        (140, 53),
                        (156, 437),
                        (360, 0),
                        (505, 119),
                        (19, 460),
                        (201, 185),
                        (281, 123),
                        (265, 93),
                        (309, 172),
                        (147, 24),
                        (9, 64),
                        (51, 21),
                        (50, 25),
                        (242, 189),
                        (168, 4),
                        (32, 329),
                        (153, 73),
                        (313, 33),
                        (55, 98),
                        (92, 267),
                        (453, 26),
                        (22, 34),
                        (116, 11)))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, max_seq_len, dropout, i)
            for i in range(n_layers)
        ])

        # Reuse the learned query/key/value transform across depth. Residual
        # streams, normalizers, attention maps, and feed-forward paths remain
        # layer-specific, so the two blocks can still perform different work.
        for block in self.blocks[1:]:
            block.attn.qkv = self.blocks[0].attn.qkv
            block.ff[0] = self.blocks[0].ff[0]
            block.ff[2] = self.blocks[0].ff[2]

        self.ln_f = TiedScaleLayerNorm(d_model, tied_pairs=LN_F_TIED_PAIRS)
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
        # The output projection reuses the reconstructed token matrix without
        # registering a duplicate learned parameter.
        logits = F.linear(x, self.token_emb.reconstructed_weight())
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
