MECHANISM: Single attention-output bias anchoring

HYPOTHESIS: Reproducing the qualified 1506-parameter nine-value-gauge and `q_bias[6]`-anchored design, then fixing one zero-initialized attention-output bias coordinate, will yield 1505 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Upgrade the current model to nine value-output rotations, anchor `q_bias[6]` as in the qualified design, and remove one reduced-coordinate bias parameter from the attention output projection.

EVIDENCE: The nine-value-gauge design with `q_bias[6]` fixed achieved 99.78% at 1506 parameters. Further query-bias, value-rotation, and LayerNorm constraints failed, motivating a distinct zero-initialized affine-bias reduction.

<<<<<<< SEARCH
class ElevenRotationEightValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and eight value-output gauges fixed."""
=======
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
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
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query, 1),
            (self.second_query + 2, 1),
        ):
=======
            (self.second_query, 0),
            (self.second_query + 1, 0),
            (self.second_query + 2, 0),
            (self.second_query, 1),
            (self.second_query + 1, 1),
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
            value_rotations[3],
            value_rotations[4],
            value_rotations[5],
            value_rotations[6],
            value_rotations[7],
        )
=======
            value_rotations[3],
            value_rotations[4],
            value_rotations[5],
            value_rotations[6],
            value_rotations[7],
            value_rotations[8],
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
            head_two_value_second_rotation,
            head_two_value_stabilizer_rotation,
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
            head_two_value_second_rotation,
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
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
class TwiceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with two bias coordinates absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 2))
        return F.linear(x, weight, bias)
=======
class AnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with one reduced bias coordinate fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationEightValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.qkv = ElevenRotationNineValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.proj = AnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 3:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[
                    self.head_dim - 3:self.head_dim - 2
                ],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 2:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationEightValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationNineValueGaugeFixedQKV):
>>>>>>> REPLACE