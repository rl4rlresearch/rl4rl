MECHANISM: Complete mean-zero input gauge fixing for unconstrained QKV rows

HYPOTHESIS: Reproducing the qualified 1494-parameter design and projecting all eleven unconstrained QKV rows onto the mean-zero basis will yield 1483 parameters while maintaining at least 99% accuracy, because fixed-scale pre-attention LayerNorm makes every removed common-mode weight component functionally invisible.

INTENDED_EDIT: Apply the qualified MLP-bias, fixed pre-attention normalization, and endpoint final-bias reductions, then store the unconstrained query, key, and value rows in the existing seven-dimensional mean-zero basis.

EVIDENCE: Reference Design 3 achieved 99.94% accuracy at 1494 parameters with all pre-attention scales fixed and already represented the last QKV row in this mean-zero basis, motivating the same exact gauge reduction for the remaining unconstrained QKV rows.

<<<<<<< SEARCH
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
        self.head_two_second_weight = nn.Parameter(
            fixed_weight[self.second_query + 1, 2:]
        )
        self.head_two_third_weight = nn.Parameter(
            fixed_weight[self.second_query + 2, 1:]
        )
        self.pre_value_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:2 * self.in_features]
        )
=======
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query] @ basis
        )
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
        self.head_two_second_weight = nn.Parameter(
            fixed_weight[self.second_query + 1, 2:]
        )
        self.head_two_third_weight = nn.Parameter(
            fixed_weight[self.second_query + 2, 1:]
        )
        self.pre_value_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:2 * self.in_features] @ basis
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_head_value_tail = nn.Parameter(
            fixed_weight[value_start + 3:value_start + self.second_query]
        )
=======
        self.first_head_value_tail = nn.Parameter(
            fixed_weight[value_start + 3:value_start + self.second_query]
            @ basis
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
            self.head_two_second_weight.copy_(
                fixed_weight[self.second_query + 1, 2:]
            )
            self.head_two_third_weight.copy_(
                fixed_weight[self.second_query + 2, 1:]
            )
            self.pre_value_weight.copy_(
                fixed_weight[self.second_query + 3:2 * self.in_features]
            )
=======
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query] @ self.basis
            )
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
            self.head_two_second_weight.copy_(
                fixed_weight[self.second_query + 1, 2:]
            )
            self.head_two_third_weight.copy_(
                fixed_weight[self.second_query + 2, 1:]
            )
            self.pre_value_weight.copy_(
                fixed_weight[self.second_query + 3:2 * self.in_features]
                @ self.basis
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.first_head_value_tail.copy_(
                fixed_weight[
                    value_start + 3:value_start + self.second_query
                ]
            )
=======
            self.first_head_value_tail.copy_(
                fixed_weight[
                    value_start + 3:value_start + self.second_query
                ] @ self.basis
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    @property
    def weight(self) -> torch.Tensor:
        first_row = F.pad(self.first_weight, (3, 0))
=======
    @property
    def weight(self) -> torch.Tensor:
        first_head_tail = (
            self.first_head_tail @ self.basis.transpose(0, 1)
        )
        pre_value_weight = (
            self.pre_value_weight @ self.basis.transpose(0, 1)
        )
        first_head_value_tail = (
            self.first_head_value_tail @ self.basis.transpose(0, 1)
        )
        first_row = F.pad(self.first_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.first_head_tail,
=======
                first_head_tail,
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.pre_value_weight,
=======
                pre_value_weight,
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.first_head_value_tail,
=======
                first_head_value_tail,
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


class EndpointBiasAnchoredLayerNorm(nn.Module):
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
        self.ln_f = EndpointBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE