MECHANISM: Complete second-head query-key Givens gauge fixing

HYPOTHESIS: Extending the qualified five-rotation 1533-parameter design with a third sequential rotation in the second attention head will produce a 1532-parameter model with at least 99% accuracy, because it removes one additional exact query-key rotational gauge while preserving initialized attention scores.

INTENDED_EDIT: Reproduce the qualified five-rotation parameterization, then rotate second-head query/key channels 2–3 and omit the resulting fixed query coefficient.

EVIDENCE: Five rotations achieved 99.52% accuracy at 1533 parameters, and every preceding incremental query-key rotation qualified; this supports completing the analogous three-rotation chain in the second head.

<<<<<<< SEARCH
class ThreeRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and three query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 3 or 2 * head_dim > in_features:
            raise ValueError("three-rotation gauge fixing requires two nontrivial heads")
=======
class SixRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and six query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("six-rotation gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_weight = nn.Parameter(fixed_weight[0, 1:])
        self.second_weight = nn.Parameter(fixed_weight[1, 1:])
        self.leading_weight = nn.Parameter(
            fixed_weight[2:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
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
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for query_start in (0, 1, self.second_query):
=======
        for query_start in (
            0,
            1,
            2,
            self.second_query,
            self.second_query + 1,
            self.second_query + 2,
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_weight.copy_(fixed_weight[0, 1:])
            self.second_weight.copy_(fixed_weight[1, 1:])
            self.leading_weight.copy_(
                fixed_weight[2:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[self.second_query + 1:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
=======
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
            self.trailing_weight.copy_(
                fixed_weight[self.second_query + 3:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (1, 0))
        second_row = F.pad(self.second_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                self.leading_weight,
                head_two_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
=======
    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (1, 0))
        second_row = F.pad(self.second_weight, (1, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (1, 0))
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
                head_two_second_row.unsqueeze(0),
                head_two_third_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ThreeRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = SixRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ThreeRotationGaugeFixedQKV):
            full_weight = module.leading_weight.new_empty(
                module.out_features, module.in_features
            )
=======
        elif isinstance(module, SixRotationGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
                module.out_features, module.in_features
            )
>>>>>>> REPLACE