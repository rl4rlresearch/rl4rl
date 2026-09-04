MECHANISM: Second attention-output bias anchoring

HYPOTHESIS: Extending the qualified 1505-parameter design by fixing a second zero-initialized reduced attention-output bias coordinate will yield 1504 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified nine-value-rotation and `q_bias[6]`-anchored model, then retain only five of the seven reduced attention-output bias coordinates.

EVIDENCE: The first attention-output bias anchor achieved 99.95% accuracy at 1505 parameters, while reductions targeting query bias, LayerNorm scale, and additional value rotations failed; anchoring the adjacent zero-initialized bias coordinate is the closest supported next reduction.

<<<<<<< SEARCH
class ElevenRotationSevenValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and seven value-output gauges fixed."""
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )
            self.initial_head_two_value_residual_rotation.copy_(
                head_two_value_residual_rotation
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
                head_two_value_residual_row.unsqueeze(0),
                head_two_value_complement_row.unsqueeze(0),
                last_row.unsqueeze(0),
=======
                head_two_value_row.unsqueeze(0),
                head_two_value_second_row.unsqueeze(0),
                head_two_value_third_row.unsqueeze(0),
                last_row.unsqueeze(0),
>>>>>>> REPLACE

<<<<<<< SEARCH
class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(out_features)
        self.weight = nn.Parameter(basis.transpose(0, 1) @ self.weight.detach())
        self.bias = nn.Parameter(basis.transpose(0, 1) @ self.bias.detach())
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.basis @ self.weight, self.basis @ self.bias)


class CausalSelfAttention(nn.Module):
=======
class MeanZeroOutputLinear(nn.Linear):
    """Linear map with its residual-stream common mode gauge-fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        basis = _mean_zero_basis(out_features)
        self.weight = nn.Parameter(basis.transpose(0, 1) @ self.weight.detach())
        self.bias = nn.Parameter(basis.transpose(0, 1) @ self.bias.detach())
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.basis @ self.weight, self.basis @ self.bias)


class TwiceAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with two reduced bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (0, 2))
        return F.linear(x, self.basis @ self.weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationSevenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.qkv = ElevenRotationNineValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
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
        elif isinstance(module, ElevenRotationSevenValueGaugeFixedQKV):
=======
        elif isinstance(module, ElevenRotationNineValueGaugeFixedQKV):
>>>>>>> REPLACE