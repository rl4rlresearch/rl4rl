MECHANISM: Complete triangular second-head value/output gauge

HYPOTHESIS: Extending the qualified 1507-parameter model with the sixth and final second-head value rotation will yield 1506 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the current four-rotation second-head parameterization with all six initialization-preserving triangular rotations, fixing two additional value-weight coordinates and compensating them in the attention output projection.

EVIDENCE: The nine-value-gauge reference achieved 99.91% accuracy at 1507 parameters; the prior attempt to test the final rotation was malformed because its search matched multiple locations, so the proposed 1506-parameter model remains untested.

<<<<<<< SEARCH
class ElevenRotationEightValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and eight value-output gauges fixed."""
=======
class ElevenRotationTenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and ten value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_rotation,
            head_two_value_second_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_complement_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
=======
            head_two_value_rotation,
            head_two_value_second_rotation,
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
            head_two_value_completion_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            "initial_head_two_value_stabilizer_rotation",
            head_two_value_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_complement_rotation",
            head_two_value_complement_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
=======
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
        self.register_buffer(
            "initial_head_two_value_completion_rotation",
            head_two_value_completion_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
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
        self.last_weight = nn.Parameter(fixed_weight[-1] @ basis)
=======
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 3:]
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
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query, 1),
            (self.second_query + 2, 1),
        ):
            row_start = value_start + value_offset
=======
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (self.second_query, 1),
            (self.second_query + 1, 1),
            (self.second_query, 2),
        ):
            row_start = value_start + value_offset
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
=======
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
            value_rotations[9],
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
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
            head_two_value_completion_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
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
            self.last_weight.copy_(fixed_weight[-1] @ self.basis)
=======
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 3:]
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.initial_head_two_value_second_rotation.copy_(
                head_two_value_second_rotation
            )
            self.initial_head_two_value_stabilizer_rotation.copy_(
                head_two_value_stabilizer_rotation
            )
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )

    @property
=======
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
            self.initial_head_two_value_completion_rotation.copy_(
                head_two_value_completion_rotation
            )

    @property
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
        last_row = self.basis @ self.last_weight
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (3, 0))
        head_two_value_second_row = F.pad(
            self.head_two_value_second_weight, (2, 0)
        )
        head_two_value_third_row = F.pad(
            self.head_two_value_third_weight, (1, 0)
        )
        last_row = self.basis @ self.last_weight
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
        self.qkv = ElevenRotationEightValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = ElevenRotationTenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            head_two_completion_columns = (
                self.proj.weight[
                    :, head_two_start:head_two_start + 2
                ]
                @ self.qkv.initial_head_two_value_completion_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_completion_columns)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationEightValueGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
=======
        elif isinstance(module, ElevenRotationTenValueGaugeFixedQKV):
            full_weight = module.first_weight.new_empty(
>>>>>>> REPLACE