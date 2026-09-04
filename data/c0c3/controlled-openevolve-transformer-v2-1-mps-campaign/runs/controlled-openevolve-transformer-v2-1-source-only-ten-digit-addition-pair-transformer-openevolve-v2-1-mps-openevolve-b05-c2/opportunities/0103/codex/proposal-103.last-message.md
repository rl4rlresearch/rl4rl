MECHANISM: Final pre-attention LayerNorm scale anchoring

HYPOTHESIS: Starting from the qualified 1497-parameter design, fixing its sole remaining pre-attention LayerNorm scale at one will yield 1496 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified MLP output-bias anchors at coordinates 0, 1, 2, and 5, then remove the remaining learned `ln1` scale so all eight pre-attention scales are fixed at one.

EVIDENCE: The 1497-parameter design achieved 99.81% accuracy, while removing output-bias coordinate 6 and hidden-bias coordinate 0 failed. All qualified designs already fix seven of eight `ln1` scales, motivating the remaining scale as an independent, initialization-preserving reduction.

<<<<<<< SEARCH
class FirstAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with reduced bias coordinate 0 fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.basis @ F.pad(self.bias, (1, 0))
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
        self.fc2 = FirstAnchoredMeanZeroOutputLinear(d_ff, d_model)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE