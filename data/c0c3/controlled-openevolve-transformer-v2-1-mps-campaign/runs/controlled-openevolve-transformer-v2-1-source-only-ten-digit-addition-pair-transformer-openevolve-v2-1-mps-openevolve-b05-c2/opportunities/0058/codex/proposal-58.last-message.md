MECHANISM: Alternate-axis residual second-head value-output gauge

HYPOTHESIS: Anchoring the residual second-head rows 5 and 7 on input column 2 will produce a 1520-parameter model with at least 99% accuracy while preserving the initialized function and all six qualified value anchors.

INTENDED_EDIT: Extend the current model with the qualified second-head complement rotation, then add a nonadjacent rows-5/7 column-two rotation and inverse output-projection compensation.

EVIDENCE: The six-value-gauge design achieved 99.89% at 1521 parameters, while the same residual rows anchored on column 1 achieved only 91.54%; changing the gauge direction tests whether that failure was anchor-direction-specific.

<<<<<<< SEARCH
class ElevenRotationFiveValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and five value-output gauges fixed."""
=======
class ElevenRotationSevenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and seven value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query + 1:-1]
        )
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 1),
            (self.second_query, 0),
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

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
            value_rotations[3],
            value_rotations[4],
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
        ) = self._gauge_fix(full_weight)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[value_start + self.second_query + 1:-1]
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        last_row = self.basis @ self.last_weight
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_two_value_row.unsqueeze(0),
                self.trailing_weight,
                last_row.unsqueeze(0),
=======
                head_two_value_row.unsqueeze(0),
                head_two_value_residual_row.unsqueeze(0),
                head_two_value_complement_row.unsqueeze(0),
                last_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationFiveValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationSevenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_columns)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationFiveValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationSevenValueGaugeFixedQKV):
>>>>>>> REPLACE