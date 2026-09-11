MECHANISM: Eleventh residual query-key Givens gauge fixing

HYPOTHESIS: Adding a first-head channels 0–1 rotation on input column 2 will produce a 1527-parameter model with at least 99% accuracy, because it preserves attention scores and all ten qualified anchors while removing one exact query coefficient.

INTENDED_EDIT: Upgrade the current six-rotation QKV parameterization to eleven sequential rotations, incorporating the qualified column-one rotations and adding a first-head column-two stabilizer rotation while preserving full-sized initialization draws.

EVIDENCE: The ten-rotation 1528-parameter design achieved 99.92% accuracy, and every preceding incremental query-key rotation qualified; the remaining first-head channels 0–1 subspace preserves the established column-zero and column-one anchors.

<<<<<<< SEARCH
class SixRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and six query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("six-rotation gauge fixing requires two suitable heads")
=======
class ElevenRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and eleven query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("eleven-rotation gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_weight = nn.Parameter(fixed_weight[0, 1:])
        self.second_weight = nn.Parameter(fixed_weight[1, 1:])
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 1:]
        )
        self.head_two_second_weight = nn.Parameter(
            fixed_weight[self.second_query + 1, 1:]
        )
        self.head_two_third_weight = nn.Parameter(
            fixed_weight[self.second_query + 2, 1:]
        )
=======
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
        self.second_weight = nn.Parameter(fixed_weight[1, 2:])
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
        self.head_two_second_weight = nn.Parameter(
            fixed_weight[self.second_query + 1, 2:]
        )
        self.head_two_third_weight = nn.Parameter(
            fixed_weight[self.second_query + 2, 1:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight = full_weight.clone()
        for query_start in (
            0,
            1,
            2,
            self.second_query,
            self.second_query + 1,
            self.second_query + 2,
        ):
            pivot = fixed_weight[query_start:query_start + 2, 0]
=======
        fixed_weight = full_weight.clone()
        rotations = (
            (0, 0),
            (1, 0),
            (2, 0),
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (0, 1),
            (self.second_query, 1),
            (1, 1),
            (self.second_query + 1, 1),
            (0, 2),
        )
        for query_start, input_coord in rotations:
            pivot = fixed_weight[
                query_start:query_start + 2, input_coord
            ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[query_start, 0] = 0.0
=======
            fixed_weight[query_start, input_coord] = 0.0
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_weight.copy_(fixed_weight[0, 1:])
            self.second_weight.copy_(fixed_weight[1, 1:])
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 1:]
            )
            self.head_two_second_weight.copy_(
                fixed_weight[self.second_query + 1, 1:]
            )
            self.head_two_third_weight.copy_(
                fixed_weight[self.second_query + 2, 1:]
            )
=======
            self.first_weight.copy_(fixed_weight[0, 3:])
            self.second_weight.copy_(fixed_weight[1, 2:])
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
            self.head_two_second_weight.copy_(
                fixed_weight[self.second_query + 1, 2:]
            )
            self.head_two_third_weight.copy_(
                fixed_weight[self.second_query + 2, 1:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_row = F.pad(self.first_weight, (1, 0))
        second_row = F.pad(self.second_weight, (1, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (1, 0))
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
=======
        first_row = F.pad(self.first_weight, (3, 0))
        second_row = F.pad(self.second_weight, (2, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (2, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (2, 0))
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SixRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, SixRotationGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationGaugeFixedQKV):
>>>>>>> REPLACE