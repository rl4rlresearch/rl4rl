MECHANISM: Tenth value-output rotation gauge fixing

HYPOTHESIS: Extending the qualified 1493-parameter design with a tenth value/output gauge rotation will produce a 1492-parameter model while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce Reference Design 3’s qualified 1493-parameter architecture, then fix second-head value-weight coordinate 2 at zero and compensate in the attention output projection.

EVIDENCE: Reference Design 3 achieved 0.9994 accuracy with 1493 parameters; the additional value rotation uses the same function-preserving construction as its nine qualified value/output gauge fixes. The previous 1492 attempt failed only because its search text did not match the supplied source.

<<<<<<< SEARCH
class ElevenRotationNineValueGaugeFixedQKV(nn.Module):
    """QKV map with eleven query-key and nine value-output gauges fixed."""
=======
class TwelveRotationTenValueGaugeFixedQKV(nn.Module):
    """QKV map with twelve query-key and ten value-output gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
=======
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
            value_rotations[7],
            value_rotations[8],
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
=======
            value_rotations[7],
            value_rotations[8],
            value_rotations[9],
        )

    def reset_from_full(self, full_weight: torch.Tensor) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
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
        bias = self.basis @ F.pad(middle_bias, (2, 2))
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
=======
        bias = self.basis @ F.pad(middle_bias, (2, 2))
        return F.linear(x, weight, bias)


class FirstThreeAndCoordinateFiveAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinates 0, 1, 2, and 5 fixed."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(
            torch.cat((full_bias[3:5], full_bias[6:]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        reduced_bias = torch.cat(
            (
                self.bias[:2],
                self.bias.new_zeros(1),
                self.bias[2:],
            )
        )
        bias = self.basis @ F.pad(reduced_bias, (3, 0))
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
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


class Block(nn.Module):
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


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FirstAndLastBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, ElevenRotationNineValueGaugeFixedQKV):
=======
        elif isinstance(module, TwelveRotationTenValueGaugeFixedQKV):
>>>>>>> REPLACE