MECHANISM: Third coupled value-output orthogonal gauge fixing

HYPOTHESIS: Completing the first-head value-column rotation chain will produce a 1524-parameter model with at least 99% accuracy while preserving the initialized model function.

INTENDED_EDIT: Reproduce the qualified eleven query-key and two value-output gauges, then rotate first-head value channels 2–3, omit the resulting fixed coefficient, and inversely rotate the matching projection columns.

EVIDENCE: The two-value-rotation design achieved 99.71% accuracy at 1525 parameters; extending the same exact symmetry is the closest supported reduction, while the alternative twelfth query-key constraint failed at 63.61%.

<<<<<<< SEARCH
class NineRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and nine query-key rotations fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("nine-rotation gauge fixing requires two suitable heads")
=======
class ElevenRotationThreeValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and three value-output gauges fixed."""
    def __init__(self, in_features: int, out_features: int, head_dim: int):
        super().__init__()
        if out_features != 3 * in_features:
            raise ValueError("gauge-fixed QKV requires out_features == 3 * in_features")
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight = self._gauge_fix(full_weight)
        self.first_weight = nn.Parameter(fixed_weight[0, 2:])
        self.second_weight = nn.Parameter(fixed_weight[1, 2:])
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
=======
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            third_value_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
            first_value_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_second_value_rotation",
            second_value_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_third_value_rotation",
            third_value_rotation.detach().clone(),
            persistent=False,
        )
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
        self.pre_value_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:2 * self.in_features]
        )
        self.first_value_weight = nn.Parameter(
            fixed_weight[2 * self.in_features, 1:]
        )
        self.second_value_weight = nn.Parameter(
            fixed_weight[2 * self.in_features + 1, 1:]
        )
        self.third_value_weight = nn.Parameter(
            fixed_weight[2 * self.in_features + 2, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[2 * self.in_features + 3:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        )
=======
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[query_start, input_coord] = 0.0
        return fixed_weight
=======
            fixed_weight[query_start, input_coord] = 0.0

        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset in (0, 1, 2):
            row_start = value_start + value_offset
            pivot = fixed_weight[row_start:row_start + 2, 0]
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
            fixed_weight[row_start:row_start + 2] = (
                rotation @ fixed_weight[row_start:row_start + 2]
            )
            fixed_weight[row_start, 0] = 0.0
            value_rotations.append(rotation)

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight = self._gauge_fix(full_weight)
        with torch.no_grad():
            self.first_weight.copy_(fixed_weight[0, 2:])
            self.second_weight.copy_(fixed_weight[1, 2:])
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
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            third_value_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
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
            self.pre_value_weight.copy_(
                fixed_weight[self.second_query + 3:2 * self.in_features]
            )
            self.first_value_weight.copy_(
                fixed_weight[2 * self.in_features, 1:]
            )
            self.second_value_weight.copy_(
                fixed_weight[2 * self.in_features + 1, 1:]
            )
            self.third_value_weight.copy_(
                fixed_weight[2 * self.in_features + 2, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[2 * self.in_features + 3:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_first_value_rotation.copy_(first_value_rotation)
            self.initial_second_value_rotation.copy_(second_value_rotation)
            self.initial_third_value_rotation.copy_(third_value_rotation)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (2, 0))
        second_row = F.pad(self.second_weight, (2, 0))
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
=======
    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (3, 0))
        second_row = F.pad(self.second_weight, (2, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (2, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (2, 0))
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        first_value_row = F.pad(self.first_value_weight, (1, 0))
        second_value_row = F.pad(self.second_value_weight, (1, 0))
        third_value_row = F.pad(self.third_value_weight, (1, 0))
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
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                third_value_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = NineRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationThreeValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def apply_initial_value_rotations(self) -> None:
        with torch.no_grad():
            first_columns = (
                self.proj.weight[:, :2]
                @ self.qkv.initial_first_value_rotation.transpose(0, 1)
            )
            self.proj.weight[:, :2].copy_(first_columns)
            second_columns = (
                self.proj.weight[:, 1:3]
                @ self.qkv.initial_second_value_rotation.transpose(0, 1)
            )
            self.proj.weight[:, 1:3].copy_(second_columns)
            third_columns = (
                self.proj.weight[:, 2:4]
                @ self.qkv.initial_third_value_rotation.transpose(0, 1)
            )
            self.proj.weight[:, 2:4].copy_(third_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        with torch.no_grad():
=======
        self.apply(self._init_weights)

        for block in self.blocks:
            block.attn.apply_initial_value_rotations()

        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, NineRotationGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationThreeValueGaugeFixedQKV):
>>>>>>> REPLACE