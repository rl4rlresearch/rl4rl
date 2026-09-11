MECHANISM: Fifth triangular first-head value/output rotation gauge

HYPOTHESIS: Reparameterizing the qualified 1507-parameter nine-value-gauge model with five triangular rotations in each attention head will yield 1506 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the first head’s four value rotations with five initialization-preserving triangular rotations, retain the qualified five second-head rotations and four query-bias anchors, and compensate every value rotation in the attention output projection.

EVIDENCE: The fifth second-head value rotation qualified at 99.91% and 1507 parameters, while adding a sixth rotation to that same head fell to 73.65%; applying the successful five-rotation structure to the first head tests the remaining asymmetric gauge without further constraining the saturated second head.

<<<<<<< SEARCH
class ElevenRotationSevenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and seven value-output gauges fixed."""
=======
class ElevenRotationTenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and ten value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
            head_two_value_complement_rotation,
            head_two_value_residual_rotation,
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
            "initial_first_value_stabilizer_rotation",
            first_value_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_first_value_complement_rotation",
            first_value_complement_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_rotation",
            head_two_value_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_complement_rotation",
            head_two_value_complement_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_residual_rotation",
            head_two_value_residual_rotation.detach().clone(),
            persistent=False,
        )
=======
        fixed_weight, value_rotations = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_value_rotations",
            value_rotations.detach().clone(),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_value_weight = nn.Parameter(
            fixed_weight[value_start, 2:]
        )
        self.second_value_weight = nn.Parameter(
            fixed_weight[value_start + 1, 1:]
        )
        self.third_value_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[value_start + 2, :1],
                    fixed_weight[value_start + 2, 2:],
                )
            )
        )
        self.first_head_value_tail = nn.Parameter(
            fixed_weight[value_start + 3:value_start + self.second_query]
        )
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 1:]
        )
        residual_row = value_start + self.second_query + 1
        self.head_two_value_residual_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[residual_row, :2],
                    fixed_weight[residual_row, 3:],
                )
            )
        )
        complement_row = value_start + self.second_query + 2
        self.head_two_value_complement_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[complement_row, :1],
                    fixed_weight[complement_row, 2:],
                )
            )
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
        self.first_value_weight = nn.Parameter(
            fixed_weight[value_start, 2:]
        )
        self.second_value_weight = nn.Parameter(
            fixed_weight[value_start + 1, 2:]
        )
        self.third_value_weight = nn.Parameter(
            fixed_weight[value_start + 2, 1:]
        )
        self.first_head_value_tail = nn.Parameter(
            fixed_weight[value_start + 3:value_start + self.second_query]
        )
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 2:]
        )
        second_row = value_start + self.second_query + 1
        self.head_two_value_second_weight = nn.Parameter(
            fixed_weight[second_row, 2:]
        )
        third_row = value_start + self.second_query + 2
        self.head_two_value_third_weight = nn.Parameter(
            fixed_weight[third_row, 1:]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 1),
            (self.second_query, 0),
            (self.second_query + 2, 1),
        ):
            row_start = value_start + value_offset
            pivot = fixed_weight[
                row_start:row_start + 2, input_coord
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
            fixed_weight[row_start:row_start + 2] = (
                rotation @ fixed_weight[row_start:row_start + 2]
            )
            fixed_weight[row_start, input_coord] = 0.0
            value_rotations.append(rotation)

        residual_rows = [
            value_start + self.second_query + 1,
            value_start + self.second_query + 3,
        ]
        pivot = fixed_weight[residual_rows, 2]
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
        fixed_weight[residual_rows] = (
            rotation @ fixed_weight[residual_rows]
        )
        fixed_weight[residual_rows[0], 2] = 0.0
        value_rotations.append(rotation)

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
            value_rotations[3],
            value_rotations[4],
            value_rotations[5],
            value_rotations[6],
        )
=======
        value_rotations = []
        value_start = 2 * self.in_features
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (self.second_query, 1),
            (self.second_query + 1, 1),
        ):
            row_start = value_start + value_offset
            pivot = fixed_weight[
                row_start:row_start + 2, input_coord
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
            fixed_weight[row_start:row_start + 2] = (
                rotation @ fixed_weight[row_start:row_start + 2]
            )
            fixed_weight[row_start, input_coord] = 0.0
            value_rotations.append(rotation)

        return fixed_weight, torch.stack(value_rotations)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
            head_two_value_complement_rotation,
            head_two_value_residual_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
        fixed_weight, value_rotations = self._gauge_fix(full_weight)
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_value_weight.copy_(
                fixed_weight[value_start, 2:]
            )
            self.second_value_weight.copy_(
                fixed_weight[value_start + 1, 1:]
            )
            self.third_value_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[value_start + 2, :1],
                        fixed_weight[value_start + 2, 2:],
                    )
                )
            )
            self.first_head_value_tail.copy_(
                fixed_weight[
                    value_start + 3:value_start + self.second_query
                ]
            )
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 1:]
            )
            residual_row = value_start + self.second_query + 1
            self.head_two_value_residual_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[residual_row, :2],
                        fixed_weight[residual_row, 3:],
                    )
                )
            )
            complement_row = value_start + self.second_query + 2
            self.head_two_value_complement_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[complement_row, :1],
                        fixed_weight[complement_row, 2:],
                    )
                )
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_first_value_rotation.copy_(first_value_rotation)
            self.initial_second_value_rotation.copy_(second_value_rotation)
            self.initial_first_value_stabilizer_rotation.copy_(
                first_value_stabilizer_rotation
            )
            self.initial_first_value_complement_rotation.copy_(
                first_value_complement_rotation
            )
            self.initial_head_two_value_rotation.copy_(
                head_two_value_rotation
            )
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )
            self.initial_head_two_value_residual_rotation.copy_(
                head_two_value_residual_rotation
            )
=======
            self.first_value_weight.copy_(
                fixed_weight[value_start, 2:]
            )
            self.second_value_weight.copy_(
                fixed_weight[value_start + 1, 2:]
            )
            self.third_value_weight.copy_(
                fixed_weight[value_start + 2, 1:]
            )
            self.first_head_value_tail.copy_(
                fixed_weight[
                    value_start + 3:value_start + self.second_query
                ]
            )
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 2:]
            )
            second_row = value_start + self.second_query + 1
            self.head_two_value_second_weight.copy_(
                fixed_weight[second_row, 2:]
            )
            third_row = value_start + self.second_query + 2
            self.head_two_value_third_weight.copy_(
                fixed_weight[third_row, 1:]
            )
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
            self.initial_value_rotations.copy_(value_rotations)
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_value_row = F.pad(self.first_value_weight, (2, 0))
        second_value_row = F.pad(self.second_value_weight, (1, 0))
        third_value_row = torch.cat(
            (
                self.third_value_weight[:1],
                self.third_value_weight.new_zeros(1),
                self.third_value_weight[1:],
            )
        )
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        head_two_value_residual_row = torch.cat(
            (
                self.head_two_value_residual_weight[:2],
                self.head_two_value_residual_weight.new_zeros(1),
                self.head_two_value_residual_weight[2:],
            )
        )
        head_two_value_complement_row = torch.cat(
            (
                self.head_two_value_complement_weight[:1],
                self.head_two_value_complement_weight.new_zeros(1),
                self.head_two_value_complement_weight[1:],
            )
        )
        last_row = self.basis @ self.last_weight
=======
        first_value_row = F.pad(self.first_value_weight, (2, 0))
        second_value_row = F.pad(self.second_value_weight, (2, 0))
        third_value_row = F.pad(self.third_value_weight, (1, 0))
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
        head_two_value_second_row = F.pad(
            self.head_two_value_second_weight, (2, 0)
        )
        head_two_value_third_row = F.pad(
            self.head_two_value_third_weight, (1, 0)
        )
        last_row = self.basis @ self.last_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                third_value_row.unsqueeze(0),
                self.first_head_value_tail,
                head_two_value_row.unsqueeze(0),
                head_two_value_residual_row.unsqueeze(0),
                head_two_value_complement_row.unsqueeze(0),
                last_row.unsqueeze(0),
=======
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                third_value_row.unsqueeze(0),
                self.first_head_value_tail,
                head_two_value_row.unsqueeze(0),
                head_two_value_second_row.unsqueeze(0),
                head_two_value_third_row.unsqueeze(0),
                last_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationSevenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationTenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            stabilized_columns = (
                self.proj.weight[:, :2]
                @ self.qkv.initial_first_value_stabilizer_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[:, :2].copy_(stabilized_columns)
            complement_columns = (
                self.proj.weight[:, 2:4]
                @ self.qkv.initial_first_value_complement_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[:, 2:4].copy_(complement_columns)
            head_two_start = self.head_dim
            head_two_columns = (
                self.proj.weight[
                    :, head_two_start:head_two_start + 2
                ]
                @ self.qkv.initial_head_two_value_rotation.transpose(0, 1)
            )
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_columns)
            head_two_complement_columns = (
                self.proj.weight[
                    :, head_two_start + 2:head_two_start + 4
                ]
                @ self.qkv.initial_head_two_value_complement_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start + 2:head_two_start + 4
            ].copy_(head_two_complement_columns)
            residual_indices = [
                head_two_start + 1,
                head_two_start + 3,
            ]
            residual_columns = (
                self.proj.weight[:, residual_indices]
                @ self.qkv.initial_head_two_value_residual_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[:, residual_indices] = residual_columns
=======
    def apply_initial_value_rotations(self) -> None:
        with torch.no_grad():
            column_starts = (
                0,
                1,
                2,
                0,
                1,
                self.head_dim,
                self.head_dim + 1,
                self.head_dim + 2,
                self.head_dim,
                self.head_dim + 1,
            )
            for rotation, column_start in zip(
                self.qkv.initial_value_rotations,
                column_starts,
            ):
                columns = (
                    self.proj.weight[:, column_start:column_start + 2]
                    @ rotation.transpose(0, 1)
                )
                self.proj.weight[
                    :, column_start:column_start + 2
                ].copy_(columns)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationSevenValueGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
=======
        elif isinstance(module, ElevenRotationTenValueGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
>>>>>>> REPLACE