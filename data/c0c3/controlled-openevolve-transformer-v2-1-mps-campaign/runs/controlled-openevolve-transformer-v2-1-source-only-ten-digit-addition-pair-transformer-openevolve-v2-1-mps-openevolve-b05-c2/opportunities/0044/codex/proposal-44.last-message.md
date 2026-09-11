MECHANISM: Mirrored per-head residual query-key orthogonal gauge fixing

HYPOTHESIS: Adding the qualified column-one stabilizer rotation to the second attention head will produce a 1530-parameter model with at least 99% accuracy because it preserves attention scores and all seven existing anchors while removing one additional query coefficient.

INTENDED_EDIT: Replace the single-rotation QKV parameterization with eight sequential Givens rotations: three column-zero anchors per head and one column-one anchor per head, preserving full-sized initialization draws.

EVIDENCE: Seven rotations achieved 99.94% accuracy at 1531 parameters, and every preceding incremental query-key rotation qualified; mirroring the seventh rotation in the previously untouched second-head stabilizer subspace is the closest supported one-parameter reduction.

<<<<<<< SEARCH
class SingleRowMeanZeroInputLinear(nn.Module):
    """QKV map with one input-null and one query-key rotation gauge fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        self.in_features = in_features
        self.out_features = out_features
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 1:])
        self.weight_rows = nn.Parameter(fixed_weight[1:-1])
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)

    def _gauge_fix(self, full_weight: torch.Tensor) -> torch.Tensor:
        fixed_weight = full_weight.clone()
        pivot = fixed_weight[:2, 0]
        radius = pivot.norm().clamp_min(torch.finfo(full_weight.dtype).tiny)
        cosine = pivot[1] / radius
        sine = -pivot[0] / radius
        rotation = torch.stack(
            (
                torch.stack((cosine, sine)),
                torch.stack((-sine, cosine)),
            )
        )
        fixed_weight[:2] = rotation @ fixed_weight[:2]
        key_start = self.in_features
        fixed_weight[key_start:key_start + 2] = (
            rotation @ fixed_weight[key_start:key_start + 2]
        )
        fixed_weight[0, 0] = 0.0
        return fixed_weight

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight = self._gauge_fix(full_weight)
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 1:])
            self.weight_rows.copy_(fixed_weight[1:-1])
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)

    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (1, 0))
        last_row = self.basis @ self.last_weight
        return torch.cat(
            (
                first_row.unsqueeze(0),
                self.weight_rows,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
=======
class EightRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and eight query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("eight-rotation gauge fixing requires two suitable heads")
        self.in_features = in_features
        self.out_features = out_features
        self.second_query = head_dim
        basis = _mean_zero_basis(in_features)
        self.register_buffer("basis", basis, persistent=False)

        full_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
        discarded_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(discarded_bias, -bound, bound)

        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 2:])
        self.second_weight = nn.Parameter(fixed_weight[1, 1:])
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
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

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight = self._gauge_fix(full_weight)
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 2:])
            self.second_weight.copy_(fixed_weight[1, 1:])
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
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

    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (2, 0))
        second_row = F.pad(self.second_weight, (1, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (2, 0))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SingleRowMeanZeroInputLinear(d_model, 3 * d_model)
=======
        self.qkv = EightRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, SingleRowMeanZeroInputLinear):
            full_weight = module.weight_rows.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
=======
        elif isinstance(module, EightRotationGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full(full_weight)
>>>>>>> REPLACE