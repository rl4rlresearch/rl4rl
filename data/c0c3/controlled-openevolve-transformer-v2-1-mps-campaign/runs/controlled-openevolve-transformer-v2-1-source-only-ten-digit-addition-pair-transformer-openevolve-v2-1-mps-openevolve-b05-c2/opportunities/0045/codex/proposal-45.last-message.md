MECHANISM: Incremental first-head column-one stabilizer rotation

HYPOTHESIS: Adding a ninth Givens rotation on first-head query/key channels 1–2 using input column 1 will produce a 1529-parameter model with at least 99% accuracy, because it preserves attention scores and all eight existing anchors while eliminating one additional query coefficient.

INTENDED_EDIT: Extend the eight-rotation QKV parameterization with a first-head `(1, 1)` stabilizer rotation and omit the resulting fixed `q_weight[1, 1]` coefficient.

EVIDENCE: Eight rotations achieved 99.84% accuracy at 1530 parameters, while every preceding incremental query-key rotation qualified; continuing the same exact symmetry is better supported than adding another LayerNorm-input row constraint, whose second application fell to 96.45%.

<<<<<<< SEARCH
class EightRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and eight query-key rotations fixed."""
=======
class NineRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and nine query-key rotations fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
            raise ValueError("eight-rotation gauge fixing requires two suitable heads")
=======
            raise ValueError("nine-rotation gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.second_weight = nn.Parameter(fixed_weight[1, 1:])
=======
        self.second_weight = nn.Parameter(fixed_weight[1, 2:])
>>>>>>> REPLACE

<<<<<<< SEARCH
            (0, 1),
            (self.second_query, 1),
        )
=======
            (0, 1),
            (self.second_query, 1),
            (1, 1),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.second_weight.copy_(fixed_weight[1, 1:])
=======
            self.second_weight.copy_(fixed_weight[1, 2:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_row = F.pad(self.second_weight, (1, 0))
=======
        second_row = F.pad(self.second_weight, (2, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = EightRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = NineRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, EightRotationGaugeFixedQKV):
=======
        elif isinstance(module, NineRotationGaugeFixedQKV):
>>>>>>> REPLACE