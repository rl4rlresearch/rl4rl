MECHANISM: Third inverse-compensated first-head query–key shear

HYPOTHESIS: Extending the qualified 1491-parameter design with a shear that fixes first-head query row 3, input coordinate 2 will yield 1490 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified MLP, normalization, twelfth query rotation, tenth value rotation, and two query shears, then remove one additional query coordinate with exact inverse-transpose key compensation.

EVIDENCE: The 1491-parameter design achieved 99.97% accuracy after two successive query–key shears, while the alternative additional value-head gauge failed at 33.71%; continuing the successful exact GL symmetry is the most informative reduction.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query rotations, three shears, and ten value gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        nn.init.uniform_(discarded_bias, -bound, bound)

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
        nn.init.uniform_(discarded_bias, -bound, bound)

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
            head_two_value_final_rotation,
        ) = self._gauge_fix(full_weight)
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
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
=======
        self.third_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[2, 1:2],
                    fixed_weight[2, 3:],
                )
            )
        )
        self.fourth_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[3, :1],
                    fixed_weight[3, 3:],
                )
            )
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 3:]
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
            (self.second_query + 1, 1),
            (0, 2),
        )
=======
            (self.second_query + 1, 1),
            (0, 2),
            (self.second_query, 2),
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

        second_shear = -fixed_weight[3, 1] / fixed_weight[2, 1]
        fixed_weight[3] = (
            fixed_weight[3] + second_shear * fixed_weight[2]
        )
        fixed_weight[key_start + 2] = (
            fixed_weight[key_start + 2]
            - second_shear * fixed_weight[key_start + 3]
        )
        fixed_weight[3, 1] = 0.0

        third_shear = -fixed_weight[3, 2] / fixed_weight[1, 2]
        fixed_weight[3] = (
            fixed_weight[3] + third_shear * fixed_weight[1]
        )
        fixed_weight[key_start + 1] = (
            fixed_weight[key_start + 1]
            - third_shear * fixed_weight[key_start + 3]
        )
        fixed_weight[3, 2] = 0.0

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
=======
            value_rotations[7],
            value_rotations[8],
            value_rotations[9],
        )

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
            head_two_value_final_rotation,
        ) = self._gauge_fix(full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
=======
            self.third_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[2, 1:2],
                        fixed_weight[2, 3:],
                    )
                )
            )
            self.fourth_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[3, :1],
                        fixed_weight[3, 3:],
                    )
                )
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 3:]
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
        first_row = F.pad(self.first_weight, (3, 0))
        second_row = F.pad(self.second_weight, (2, 0))
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (2, 0))
=======
        first_row = F.pad(self.first_weight, (3, 0))
        second_row = F.pad(self.second_weight, (2, 0))
        third_row = torch.cat(
            (
                self.third_weight.new_zeros(1),
                self.third_weight[:1],
                self.third_weight.new_zeros(1),
                self.third_weight[1:],
            )
        )
        fourth_row = torch.cat(
            (
                self.fourth_weight[:1],
                self.fourth_weight.new_zeros(2),
                self.fourth_weight[1:],
            )
        )
        head_two_row = F.pad(self.head_two_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
=======
                third_row.unsqueeze(0),
                fourth_row.unsqueeze(0),
                head_two_row.unsqueeze(0),
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

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, 2, and 5 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(
            torch.cat((full_bias[3:5], full_bias[6:]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced_bias = torch.cat(
            (
                self.bias[:2],
                self.bias.new_zeros(1),
                self.bias[2:],
            )
        )
        bias = self.basis @ F.pad(reduced_bias, (3, 0))
        return F.linear(x, self.basis @ self.weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = MeanZeroOutputLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(
            d_ff, d_model
        )
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
class SeptupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with seven scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 7), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class FixedScaleLayerNorm(nn.Module):
    """LayerNorm with all scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None, self.eps)


class FirstBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinate 0 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 0))
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = FirstBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE