MECHANISM: Residual second-head column-one value-output gauge

HYPOTHESIS: Rotating second-head value rows 5 and 7 on input column 1 will produce a 1520-parameter model with at least 99% accuracy while preserving the initialized function and all six qualified value anchors.

INTENDED_EDIT: Add a seventh, nonadjacent value-output Givens rotation, omit the resulting fixed `value_weight[5, 1]` coefficient, and inversely rotate attention-output columns 5 and 7.

EVIDENCE: The mirrored complement rotation qualified at 99.89% with 1521 parameters, while the row-5–6 column-zero extension fell to 80.13%; using the residual rows 5 and 7 on the successful column-one direction avoids disturbing either qualified second-head zero.

<<<<<<< SEARCH
class ElevenRotationSixValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and six value-output gauges fixed."""
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
            head_two_value_complement_rotation,
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
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
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
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_value_middle = nn.Parameter(
            fixed_weight[
                value_start + self.second_query + 1:
                value_start + self.second_query + 2
            ]
        )
=======
        middle_row = value_start + self.second_query + 1
        self.head_two_value_middle = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[middle_row:middle_row + 1, :1],
                    fixed_weight[middle_row:middle_row + 1, 2:],
                ),
                dim=1,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[row_start, input_coord] = 0.0
            value_rotations.append(rotation)

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
            value_rotations[3],
            value_rotations[4],
            value_rotations[5],
        )
=======
            fixed_weight[row_start, input_coord] = 0.0
            value_rotations.append(rotation)

        residual_rows = [
            value_start + self.second_query + 1,
            value_start + self.second_query + 3,
        ]
        pivot = fixed_weight[residual_rows, 1]
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
        fixed_weight[residual_rows[0], 1] = 0.0
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
            head_two_value_complement_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
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
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_value_middle.copy_(
                fixed_weight[
                    value_start + self.second_query + 1:
                    value_start + self.second_query + 2
                ]
            )
=======
            middle_row = value_start + self.second_query + 1
            self.head_two_value_middle.copy_(
                torch.cat(
                    (
                        fixed_weight[middle_row:middle_row + 1, :1],
                        fixed_weight[middle_row:middle_row + 1, 2:],
                    ),
                    dim=1,
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )
=======
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )
            self.initial_head_two_value_residual_rotation.copy_(
                head_two_value_residual_rotation
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        head_two_value_complement_row = torch.cat(
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (1, 0))
        head_two_value_middle_row = torch.cat(
            (
                self.head_two_value_middle[:, :1],
                self.head_two_value_middle.new_zeros((1, 1)),
                self.head_two_value_middle[:, 1:],
            ),
            dim=1,
        )
        head_two_value_complement_row = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_two_value_row.unsqueeze(0),
                self.head_two_value_middle,
                head_two_value_complement_row.unsqueeze(0),
=======
                head_two_value_row.unsqueeze(0),
                head_two_value_middle_row,
                head_two_value_complement_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationSixValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationSevenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.proj.weight[
                :, head_two_start + 2:head_two_start + 4
            ].copy_(head_two_complement_columns)
=======
            self.proj.weight[
                :, head_two_start + 2:head_two_start + 4
            ].copy_(head_two_complement_columns)
            residual_indices = [
                head_two_start + 1,
                head_two_start + 3,
            ]
            head_two_residual_columns = (
                self.proj.weight[:, residual_indices]
                @ self.qkv.initial_head_two_value_residual_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, residual_indices
            ] = head_two_residual_columns
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationSixValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationSevenValueGaugeFixedQKV):
>>>>>>> REPLACE