MECHANISM: First-head query-key unit-shear gauge

HYPOTHESIS: Reproducing the qualified 1507-parameter nine-value-gauge model and eliminating query-weight coordinate `(3, 1)` through an initialization-preserving first-head query/key shear will yield 1506 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Upgrade the current value/output parameterization to the qualified nine-rotation design, then apply a unit lower-triangular query-key shear that fixes one additional first-head query coordinate without fixing its query bias or a weight magnitude.

EVIDENCE: The nine-value-gauge design achieved 99.91% at 1507 parameters. The failed 1506 attempts constrained the sensitive second head or fixed a nonzero scale pivot; this instead uses an exact first-head GL shear, where the already-qualified zero `q_bias[2]` makes the bias compatible with the gauge.

<<<<<<< SEARCH
class ElevenRotationEightValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and eight value-output gauges fixed."""
=======
class ElevenRotationOneShearNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven rotations, one query-key shear, and nine value gauges."""
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
            head_two_value_stabilizer_rotation,
            head_two_value_complement_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "initial_head_two_value_second_rotation",
            head_two_value_second_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_stabilizer_rotation",
            head_two_value_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_complement_rotation",
            head_two_value_complement_rotation.detach().clone(),
            persistent=False,
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
=======
        first_head_tail = fixed_weight[3:self.second_query].flatten()
        self.first_head_tail = nn.Parameter(
            torch.cat((first_head_tail[:1], first_head_tail[2:]))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 2:]
        )
        second_row = value_start + self.second_query + 1
        self.head_two_value_second_weight = nn.Parameter(
            fixed_weight[second_row, 1:]
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[query_start, input_coord] = 0.0

        value_rotations = []
        value_start = 2 * self.in_features
=======
            fixed_weight[query_start, input_coord] = 0.0

        shear_source = 2
        shear_target = 3
        shear_coord = 1
        shear_pivot = fixed_weight[shear_source, shear_coord]
        tiny = torch.finfo(full_weight.dtype).tiny
        safe_pivot = torch.where(
            shear_pivot >= 0,
            shear_pivot.clamp_min(tiny),
            shear_pivot.clamp_max(-tiny),
        )
        shear = fixed_weight[shear_target, shear_coord] / safe_pivot
        fixed_weight[shear_target] = (
            fixed_weight[shear_target] - shear * fixed_weight[shear_source]
        )
        key_source = self.in_features + shear_source
        key_target = self.in_features + shear_target
        fixed_weight[key_source] = (
            fixed_weight[key_source] + shear * fixed_weight[key_target]
        )
        fixed_weight[shear_target, shear_coord] = 0.0

        value_rotations = []
        value_start = 2 * self.in_features
>>>>>>> REPLACE

<<<<<<< SEARCH
        for value_offset, input_coord in (
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 1),
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query, 1),
            (self.second_query + 2, 1),
        ):
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            value_rotations[6],
            value_rotations[7],
        )
=======
            value_rotations[6],
            value_rotations[7],
            value_rotations[8],
        )
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
            head_two_value_second_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_complement_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
    def reset_from_full(self, full_weight: torch.Tensor) -> None:
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
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
=======
            first_head_tail = fixed_weight[3:self.second_query].flatten()
            self.first_head_tail.copy_(
                torch.cat((first_head_tail[:1], first_head_tail[2:]))
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 2:]
            )
            second_row = value_start + self.second_query + 1
            self.head_two_value_second_weight.copy_(
                fixed_weight[second_row, 1:]
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.initial_head_two_value_rotation.copy_(
                head_two_value_rotation
            )
            self.initial_head_two_value_second_rotation.copy_(
                head_two_value_second_rotation
            )
            self.initial_head_two_value_stabilizer_rotation.copy_(
                head_two_value_stabilizer_rotation
            )
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (2, 0))
=======
        third_row = F.pad(self.third_weight, (1, 0))
        first_head_tail = torch.cat(
            (
                self.first_head_tail[:1],
                self.first_head_tail.new_zeros(1),
                self.first_head_tail[1:],
            )
        ).view(self.second_query - 3, self.in_features)
        head_two_row = F.pad(self.head_two_weight, (2, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
        head_two_value_second_row = F.pad(
            self.head_two_value_second_weight, (1, 0)
        )
        head_two_value_complement_row = torch.cat(
            (
                self.head_two_value_complement_weight[:1],
                self.head_two_value_complement_weight.new_zeros(1),
                self.head_two_value_complement_weight[1:],
            )
        )
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
        head_two_value_second_row = F.pad(
            self.head_two_value_second_weight, (2, 0)
        )
        head_two_value_third_row = F.pad(
            self.head_two_value_third_weight, (1, 0)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
=======
                third_row.unsqueeze(0),
                first_head_tail,
                head_two_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_two_value_row.unsqueeze(0),
                head_two_value_second_row.unsqueeze(0),
                head_two_value_complement_row.unsqueeze(0),
                last_row.unsqueeze(0),
=======
                head_two_value_row.unsqueeze(0),
                head_two_value_second_row.unsqueeze(0),
                head_two_value_third_row.unsqueeze(0),
                last_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationEightValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationOneShearNineValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationEightValueGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
=======
        elif isinstance(module, ElevenRotationOneShearNineValueGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
>>>>>>> REPLACE