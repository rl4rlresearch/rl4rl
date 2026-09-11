MECHANISM: Residual query-key orthogonal gauge fixing

HYPOTHESIS: Adding a seventh Givens rotation on first-head query/key channels 0–1 using input column 1 will produce a 1531-parameter model with at least 99% accuracy, because it preserves attention scores and the six existing column-zero anchors while eliminating one additional query coefficient.

INTENDED_EDIT: Reproduce the qualified six-rotation QKV parameterization, apply one stabilizer rotation within the first head, and omit the resulting fixed `q_weight[0, 1]` coefficient while preserving full-sized initialization draws.

EVIDENCE: The six-rotation 1532-parameter design achieved 99.97% accuracy, and every preceding incremental query-key rotation qualified; this supports extending the same exact symmetry into the remaining first-head subspace.

<<<<<<< SEARCH
class TwoHeadGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and two query-key rotation gauges fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 2 or 2 * head_dim > in_features:
            raise ValueError("two-head gauge fixing requires two nontrivial heads")
=======
class SevenRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and seven query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("seven-rotation gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 1:])
        self.leading_weight = nn.Parameter(fixed_weight[1:self.second_query])
        self.second_weight = nn.Parameter(
            fixed_weight[self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 2:])
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
        fixed_weight = full_weight.clone()
        for query_start in (0, self.second_query):
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
        fixed_weight = full_weight.clone()
        rotations = (
            (0, 0),
            (1, 0),
            (2, 0),
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (0, 1),
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
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 1:])
            self.leading_weight.copy_(
                fixed_weight[1:self.second_query]
            )
            self.second_weight.copy_(
                fixed_weight[self.second_query, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[self.second_query + 1:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
=======
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 2:])
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
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                self.leading_weight,
                second_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
=======
    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (2, 0))
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
        self.qkv = TwoHeadGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = SevenRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, TwoHeadGaugeFixedQKV):
            full_weight = module.leading_weight.new_empty(
                module.out_features, module.in_features
            )
=======
        elif isinstance(module, SevenRotationGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
                module.out_features, module.in_features
            )
>>>>>>> REPLACE