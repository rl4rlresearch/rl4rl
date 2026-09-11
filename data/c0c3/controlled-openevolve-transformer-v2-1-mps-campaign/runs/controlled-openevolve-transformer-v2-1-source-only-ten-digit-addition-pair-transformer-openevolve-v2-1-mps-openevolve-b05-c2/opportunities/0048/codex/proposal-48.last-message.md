MECHANISM: Complete per-head triangular query-key gauge fixing

HYPOTHESIS: Completing the second head’s six-dimensional orthogonal gauge will produce a 1526-parameter model with at least 99% accuracy because the added rotation preserves attention scores while removing one exact query coefficient.

INTENDED_EDIT: Upgrade the five-rotation QKV parameterization to twelve rotations, including all eleven qualified anchors plus the mirrored second-head column-two rotation, while preserving full-sized initialization draws.

EVIDENCE: The eleven-rotation design achieved 99.92% accuracy at 1527 parameters, and every preceding incremental query-key rotation qualified; the remaining second-head channels 0–1 rotation is the exact mirror of its qualified first-head counterpart.

<<<<<<< SEARCH
class FiveRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and five query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("five-rotation gauge fixing requires two suitable heads")
=======
class TwelveRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and twelve query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("twelve-rotation gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight = self._gauge_fix(full_weight)
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
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 2:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
        self.second_weight = nn.Parameter(fixed_weight[1, 2:])
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 3:]
        )
        self.head_two_second_weight = nn.Parameter(
            fixed_weight[self.second_query + 1, 2:]
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
    def _gauge_fix(self, full_weight: torch.Tensor) -> torch.Tensor:
        fixed_weight = full_weight.clone()
        for query_start in (0, 1, 2, self.second_query, self.second_query + 1):
            pivot = fixed_weight[query_start:query_start + 2, 0]
            radius = pivot.norm().clamp_min(
                torch.finfo(full_weight.dtype).tiny
            )
            cosine = pivot[1] / radius
            sine = -pivot[0] / radius
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            fixed_weight[query_start:query_start + 2] = (
                rotation @ fixed_weight[query_start:query_start + 2]
            )
            key_start = self.in_features + query_start
            fixed_weight[key_start:key_start + 2] = (
                rotation @ fixed_weight[key_start:key_start + 2]
            )
            fixed_weight[query_start, 0] = 0.0
        return fixed_weight
=======
    def _gauge_fix(self, full_weight: torch.Tensor) -> torch.Tensor:
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
            (self.second_query, 2),
        )
        for query_start, input_coord in rotations:
            pivot = fixed_weight[
                query_start:query_start + 2, input_coord
            ]
            radius = pivot.norm().clamp_min(
                torch.finfo(full_weight.dtype).tiny
            )
            cosine = pivot[1] / radius
            sine = -pivot[0] / radius
            rotation = torch.stack(
                (
                    torch.stack((cosine, sine)),
                    torch.stack((-sine, cosine)),
                )
            )
            fixed_weight[query_start:query_start + 2] = (
                rotation @ fixed_weight[query_start:query_start + 2]
            )
            key_start = self.in_features + query_start
            fixed_weight[key_start:key_start + 2] = (
                rotation @ fixed_weight[key_start:key_start + 2]
            )
            fixed_weight[query_start, input_coord] = 0.0
        return fixed_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight = self._gauge_fix(full_weight)
        with torch.no_grad():
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
            self.trailing_weight.copy_(
                fixed_weight[self.second_query + 2:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight = self._gauge_fix(full_weight)
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 3:])
            self.second_weight.copy_(fixed_weight[1, 2:])
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 3:]
            )
            self.head_two_second_weight.copy_(
                fixed_weight[self.second_query + 1, 2:]
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
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                second_row.unsqueeze(0),
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
                head_two_second_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
=======
    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (3, 0))
        second_row = F.pad(self.second_weight, (2, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (3, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (2, 0))
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
        self.qkv = FiveRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = TwelveRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FiveRotationGaugeFixedQKV):
=======
        elif isinstance(module, TwelveRotationGaugeFixedQKV):
>>>>>>> REPLACE