MECHANISM: Coupled value-output orthogonal gauge fixing

HYPOTHESIS: Fixing one first-head value coefficient through a shared rotation of the value rows and output-projection columns will produce a 1526-parameter model with at least 99% accuracy while preserving the initialized model function.

INTENDED_EDIT: Retain the qualified eleven query-key rotations, rotate first-head value channels 0–1 to eliminate `value_weight[0, 0]`, omit that coefficient, and apply the inverse rotation to the corresponding attention-output projection columns.

EVIDENCE: The 1527-parameter eleven-rotation design achieved 99.92%, whereas extending the second head to twelve query-key anchors fell to 63.61%; this tests a distinct exact value-output factorization symmetry without adding the failed query constraint.

<<<<<<< SEARCH
class ElevenRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and eleven query-key rotations fixed."""
=======
class ElevenRotationGaugeFixedQKV(nn.Module):
    """QKV map with input, query-key, and one value-output gauge fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_weight = self._gauge_fix(full_weight)
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
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
        fixed_weight, value_rotation = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_value_rotation",
            value_rotation.detach().clone(),
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
        self.trailing_weight = nn.Parameter(
            fixed_weight[2 * self.in_features + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _gauge_fix(self, full_weight: torch.Tensor) -> torch.Tensor:
=======
    def _gauge_fix(
        self, full_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[query_start, input_coord] = 0.0
        return fixed_weight
=======
            fixed_weight[query_start, input_coord] = 0.0

        value_start = 2 * self.in_features
        pivot = fixed_weight[value_start:value_start + 2, 0]
        radius = pivot.norm().clamp_min(
            torch.finfo(full_weight.dtype).tiny
        )
        cosine = pivot[1] / radius
        sine = -pivot[0] / radius
        value_rotation = torch.stack(
            (
                torch.stack((cosine, sine)),
                torch.stack((-sine, cosine)),
            )
        )
        fixed_weight[value_start:value_start + 2] = (
            value_rotation @ fixed_weight[value_start:value_start + 2]
        )
        fixed_weight[value_start, 0] = 0.0
        return fixed_weight, value_rotation
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                fixed_weight[self.second_query, 2:]
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
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight, value_rotation = self._gauge_fix(full_weight)
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
            self.trailing_weight.copy_(
                fixed_weight[2 * self.in_features + 1:-1]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_value_rotation.copy_(value_rotation)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        first_value_row = F.pad(self.first_value_weight, (1, 0))
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
                self.trailing_weight,
                last_row.unsqueeze(0),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        self.register_buffer("mask", mask, persistent=False)

    def apply_initial_value_rotation(self) -> None:
        with torch.no_grad():
            rotated_columns = (
                self.proj.weight[:, :2]
                @ self.qkv.initial_value_rotation.transpose(0, 1)
            )
            self.proj.weight[:, :2].copy_(rotated_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        with torch.no_grad():
=======
        self.apply(self._init_weights)

        for block in self.blocks:
            block.attn.apply_initial_value_rotation()

        with torch.no_grad():
>>>>>>> REPLACE