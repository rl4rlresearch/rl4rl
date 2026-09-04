MECHANISM: Mirrored second-head column-one query-key stabilizer rotation

HYPOTHESIS: Extending the qualified nine-rotation 1529-parameter design with a second-head channels 1–2 rotation on input column 1 will produce a 1528-parameter model with at least 99% accuracy, because it preserves attention scores and all nine existing anchors while removing one exact query coefficient.

INTENDED_EDIT: Upgrade the current QKV parameterization to ten sequential Givens rotations, incorporating the qualified eighth and ninth rotations and adding the mirrored second-head tenth rotation while preserving full-sized initialization draws.

EVIDENCE: Nine rotations achieved 99.93% accuracy at 1529 parameters, and the earlier mirrored second-head stabilizer achieved 99.84%; alternating the established column-one rotation chain into the second head is the closest supported one-parameter reduction.

<<<<<<< SEARCH
class SevenRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and seven query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("seven-rotation gauge fixing requires two suitable heads")
=======
class TenRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and ten query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("ten-rotation gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_weight = nn.Parameter(fixed_weight[0, 2:])
        self.second_weight = nn.Parameter(fixed_weight[1, 1:])
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
=======
        self.first_weight = nn.Parameter(fixed_weight[0, 2:])
        self.second_weight = nn.Parameter(fixed_weight[1, 2:])
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        rotations = (
            (0, 0),
            (1, 0),
            (2, 0),
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (0, 1),
        )
=======
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
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_weight.copy_(fixed_weight[0, 2:])
            self.second_weight.copy_(fixed_weight[1, 1:])
            self.third_weight.copy_(fixed_weight[2, 1:])
=======
            self.first_weight.copy_(fixed_weight[0, 2:])
            self.second_weight.copy_(fixed_weight[1, 2:])
            self.third_weight.copy_(fixed_weight[2, 1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        first_row = F.pad(self.first_weight, (2, 0))
        second_row = F.pad(self.second_weight, (1, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (1, 0))
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
=======
        first_row = F.pad(self.first_weight, (2, 0))
        second_row = F.pad(self.second_weight, (2, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (2, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (2, 0))
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SevenRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = TenRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, SevenRotationGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
=======
        elif isinstance(module, TenRotationGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
>>>>>>> REPLACE