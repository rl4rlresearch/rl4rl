MECHANISM: Complete mean-zero value projection with full per-head orthogonal gauge fixing

HYPOTHESIS: Projecting every value row onto the LayerNorm-visible mean-zero subspace and extending value/output gauge fixing from nine to twelve rotations will reduce the qualified model from 1483 to 1474 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Represent each head’s four value rows as a mean-zero triangular matrix with 22 learned coordinates, and compensate all six per-head rotations in the attention output projection.

EVIDENCE: The current 1483-parameter model achieved 99.99% accuracy after mean-zero projection removed functionally invisible QKV common modes; the same fixed-scale pre-attention LayerNorm makes the remaining value-row common modes invisible, while the existing successful value/output rotations motivate completing the six-dimensional orthogonal gauge of each four-dimensional head.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
class ElevenRotationTwelveValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and twelve value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
            head_two_value_second_rotation,
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
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
            "initial_head_two_value_second_rotation",
            head_two_value_second_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_third_rotation",
            head_two_value_third_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_stabilizer_rotation",
            head_two_value_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_second_stabilizer_rotation",
            head_two_value_second_stabilizer_rotation.detach().clone(),
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
        value_start = 2 * self.in_features
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
            @ basis
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
=======
        value_start = 2 * self.in_features
        self.value_weights = nn.ParameterList(
            [
                nn.Parameter(
                    fixed_weight[
                        value_start + head_start + row,
                        leading_zeros:-1,
                    ]
                )
                for head_start in (0, self.second_query)
                for row, leading_zeros in enumerate((3, 2, 1, 0))
            ]
        )
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

        return (
            fixed_weight,
            value_rotations[0],
            value_rotations[1],
            value_rotations[2],
            value_rotations[3],
            value_rotations[4],
            value_rotations[5],
            value_rotations[6],
            value_rotations[7],
            value_rotations[8],
        )
=======
        value_rotations = []
        value_start = 2 * self.in_features
        value_rows = fixed_weight[value_start:]
        fixed_weight[value_start:] = (
            value_rows - value_rows.mean(dim=1, keepdim=True)
        )
        for head_start in (0, self.second_query):
            for value_offset, input_coord in (
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (0, 2),
            ):
                row_start = value_start + head_start + value_offset
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
        (
            fixed_weight,
            first_value_rotation,
            second_value_rotation,
            first_value_stabilizer_rotation,
            first_value_complement_rotation,
            head_two_value_rotation,
            head_two_value_second_rotation,
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
=======
        fixed_weight, value_rotations = self._gauge_fix(full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            value_start = 2 * self.in_features
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
                ] @ self.basis
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
=======
            value_start = 2 * self.in_features
            value_slices = (
                fixed_weight[
                    value_start + head_start + row,
                    leading_zeros:-1,
                ]
                for head_start in (0, self.second_query)
                for row, leading_zeros in enumerate((3, 2, 1, 0))
            )
            for parameter, value_slice in zip(
                self.value_weights, value_slices
            ):
                parameter.copy_(value_slice)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            self.initial_head_two_value_second_rotation.copy_(
                head_two_value_second_rotation
            )
            self.initial_head_two_value_third_rotation.copy_(
                head_two_value_third_rotation
            )
            self.initial_head_two_value_stabilizer_rotation.copy_(
                head_two_value_stabilizer_rotation
            )
            self.initial_head_two_value_second_stabilizer_rotation.copy_(
                head_two_value_second_stabilizer_rotation
            )
=======
            self.initial_value_rotations.copy_(value_rotations)
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_value_tail = (
            self.first_head_value_tail @ self.basis.transpose(0, 1)
        )
=======
        value_rows = []
        for parameter, leading_zeros in zip(
            self.value_weights, (3, 2, 1, 0) * 2
        ):
            row_tail = torch.cat(
                (parameter, -parameter.sum().reshape(1))
            )
            value_rows.append(F.pad(row_tail, (leading_zeros, 0)))
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
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
        head_two_value_second_row = F.pad(
            self.head_two_value_second_weight, (2, 0)
        )
        head_two_value_third_row = F.pad(
            self.head_two_value_third_weight, (1, 0)
        )
        last_row = self.basis @ self.last_weight
        return torch.cat(
=======
        return torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                first_value_row.unsqueeze(0),
                second_value_row.unsqueeze(0),
                third_value_row.unsqueeze(0),
                first_head_value_tail,
                head_two_value_row.unsqueeze(0),
                head_two_value_second_row.unsqueeze(0),
                head_two_value_third_row.unsqueeze(0),
                last_row.unsqueeze(0),
=======
                torch.stack(value_rows, dim=0),
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationNineValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationTwelveValueGaugeFixedQKV(
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
            head_two_second_columns = (
                self.proj.weight[
                    :, head_two_start + 1:head_two_start + 3
                ]
                @ self.qkv.initial_head_two_value_second_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start + 1:head_two_start + 3
            ].copy_(head_two_second_columns)
            head_two_third_columns = (
                self.proj.weight[
                    :, head_two_start + 2:head_two_start + 4
                ]
                @ self.qkv.initial_head_two_value_third_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start + 2:head_two_start + 4
            ].copy_(head_two_third_columns)
            head_two_stabilized_columns = (
                self.proj.weight[
                    :, head_two_start:head_two_start + 2
                ]
                @ self.qkv.initial_head_two_value_stabilizer_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_stabilized_columns)
            head_two_second_stabilized_columns = (
                self.proj.weight[
                    :, head_two_start + 1:head_two_start + 3
                ]
                @ self.qkv.initial_head_two_value_second_stabilizer_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start + 1:head_two_start + 3
            ].copy_(head_two_second_stabilized_columns)
=======
    def apply_initial_value_rotations(self) -> None:
        starts = (
            0,
            1,
            2,
            0,
            1,
            0,
            self.head_dim,
            self.head_dim + 1,
            self.head_dim + 2,
            self.head_dim,
            self.head_dim + 1,
            self.head_dim,
        )
        with torch.no_grad():
            for start, rotation in zip(
                starts, self.qkv.initial_value_rotations
            ):
                columns = (
                    self.proj.weight[:, start:start + 2]
                    @ rotation.transpose(0, 1)
                )
                self.proj.weight[:, start:start + 2].copy_(columns)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationNineValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationTwelveValueGaugeFixedQKV):
>>>>>>> REPLACE