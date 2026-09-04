MECHANISM: Inverse-compensated query–key shear gauge fixing

HYPOTHESIS: Reproducing the qualified 1493-parameter ten-value-gauge design and fixing first-head query row 2, input coordinate 2 through an exact query–key shear will yield 1492 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Add the qualified tenth second-head value rotation with output-projection compensation, then remove one first-head query coordinate using a triangular shear and inverse-transpose key compensation.

EVIDENCE: The ten-value-gauge reference achieved 99.97% accuracy at 1493 parameters; the prior 1492 value-head recanonicalization failed, while the independent query–key shear attempt was not tested because its patch did not apply.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query rotations, one shear, and ten value gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
=======
            head_two_value_second_stabilizer_rotation,
            head_two_value_final_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "initial_head_two_value_second_stabilizer_rotation",
            head_two_value_second_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
=======
        self.register_buffer(
            "initial_head_two_value_second_stabilizer_rotation",
            head_two_value_second_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_final_rotation",
            head_two_value_final_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
=======
        self.third_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[2, 1:2],
                    fixed_weight[2, 3:],
                )
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 2:]
        )
=======
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 3:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[query_start, input_coord] = 0.0

        value_rotations = []
=======
            fixed_weight[query_start, input_coord] = 0.0

        shear = -fixed_weight[2, 2] / fixed_weight[1, 2]
        fixed_weight[2] = fixed_weight[2] + shear * fixed_weight[1]
        key_start = self.in_features
        fixed_weight[key_start + 1] = (
            fixed_weight[key_start + 1]
            - shear * fixed_weight[key_start + 2]
        )
        fixed_weight[2, 2] = 0.0

        value_rotations = []
>>>>>>> REPLACE

<<<<<<< SEARCH
            (self.second_query, 1),
            (self.second_query + 1, 1),
        ):
=======
            (self.second_query, 1),
            (self.second_query + 1, 1),
            (self.second_query, 2),
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
            value_rotations[7],
            value_rotations[8],
        )
=======
            value_rotations[7],
            value_rotations[8],
            value_rotations[9],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
            head_two_value_second_stabilizer_rotation,
            head_two_value_final_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.third_weight.copy_(fixed_weight[2, 1:])
=======
            self.third_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[2, 1:2],
                        fixed_weight[2, 3:],
                    )
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 2:]
            )
=======
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 3:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.initial_head_two_value_second_stabilizer_rotation.copy_(
                head_two_value_second_stabilizer_rotation
            )

    @property
=======
            self.initial_head_two_value_second_stabilizer_rotation.copy_(
                head_two_value_second_stabilizer_rotation
            )
            self.initial_head_two_value_final_rotation.copy_(
                head_two_value_final_rotation
            )

    @property
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_row = F.pad(self.third_weight, (1, 0))
=======
        third_row = torch.cat(
            (
                self.third_weight.new_zeros(1),
                self.third_weight[:1],
                self.third_weight.new_zeros(1),
                self.third_weight[1:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.proj.weight[
                :, head_two_start + 1:head_two_start + 3
            ].copy_(head_two_second_stabilized_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
            self.proj.weight[
                :, head_two_start + 1:head_two_start + 3
            ].copy_(head_two_second_stabilized_columns)
            head_two_final_columns = (
                self.proj.weight[
                    :, head_two_start:head_two_start + 2
                ]
                @ self.qkv.initial_head_two_value_final_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_final_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE