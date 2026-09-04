MECHANISM: Nonadjacent final-normalization bias anchoring

HYPOTHESIS: Fixing final LayerNorm bias coordinate 2 while preserving coordinate 1 will reduce the qualified 1495-parameter design to 1494 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified 1495-parameter MLP and pre-attention reductions, then reconstruct final LayerNorm bias coordinates 0 and 2 as zero.

EVIDENCE: Fixing final LayerNorm bias coordinate 0 achieved 99.83% accuracy at 1495 parameters, whereas additionally fixing coordinate 1 fell to 71.26%; testing coordinate 2 distinguishes coordinate sensitivity from a general 1494-parameter capacity limit.

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


class FirstAndThirdBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinates 0 and 2 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (
                self.bias.new_zeros(1),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
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
        self.ln_f = FirstAndThirdBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE