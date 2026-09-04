MECHANISM: Remaining second-head value-output rotation gauge fixing

HYPOTHESIS: Reproducing the qualified 1493-parameter design and fixing the sixth orthogonal value-output rotation in the second attention head will yield 1492 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Apply all qualified 1493-parameter reductions, then add a sixth second-head value rotation, fix the resulting QKV value-weight coordinate, and compensate the attention projection at initialization.

EVIDENCE: The 1493-parameter design achieved 99.94% accuracy after completing the second-head query-key rotation gauge; its value-output parameterization fixes only five of the six corresponding second-head orthogonal rotations, motivating an exact, initialization-preserving one-parameter reduction.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
class TwelveRotationTenValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query-key and ten value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
=======
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
            head_two_value_complement_rotation,
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
            "initial_head_two_value_complement_rotation",
            head_two_value_complement_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
=======
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
            (1, 1),
            (self.second_query + 1, 1),
            (0, 2),
        )
=======
            (1, 1),
            (self.second_query + 1, 1),
            (0, 2),
            (self.second_query, 2),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (self.second_query + 2, 0),
            (self.second_query, 1),
            (self.second_query + 1, 1),
        ):
=======
            (self.second_query + 2, 0),
            (self.second_query, 1),
            (self.second_query + 1, 1),
            (self.second_query, 2),
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
            value_rotations[6],
            value_rotations[7],
            value_rotations[8],
        )
=======
            value_rotations[6],
            value_rotations[7],
            value_rotations[8],
            value_rotations[9],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
            head_two_value_third_rotation,
            head_two_value_stabilizer_rotation,
            head_two_value_second_stabilizer_rotation,
            head_two_value_complement_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
=======
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
            self.initial_head_two_value_complement_rotation.copy_(
                head_two_value_complement_rotation
            )

    @property
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_row = F.pad(self.head_two_weight, (2, 0))
=======
        head_two_row = F.pad(self.head_two_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
class FirstThreeAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, and 2 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[3:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (3, 0))
        return F.linear(x, self.basis @ self.weight, bias)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = ElevenRotationNineValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = TwelveRotationTenValueGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
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
            head_two_complement_columns = (
                self.proj.weight[
                    :, head_two_start:head_two_start + 2
                ]
                @ self.qkv.initial_head_two_value_complement_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_complement_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = FirstThreeAnchoredMeanZeroOutputLinear(d_ff, d_model)
=======
        self.fc2 = FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(
            d_ff, d_model
        )
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


class FirstAndLastBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinates 0 and 7 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 1))
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
        self.ln_f = FirstAndLastBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationNineValueGaugeFixedQKV):
=======
        elif isinstance(module, TwelveRotationTenValueGaugeFixedQKV):
>>>>>>> REPLACE