MECHANISM: Nonadjacent final-normalization bias anchoring

HYPOTHESIS: Reproducing the qualified 1495-parameter design and fixing final LayerNorm bias coordinate 3 while preserving coordinates 1 and 2 will yield 1494 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Restore the qualified MLP output-bias and fixed pre-attention normalization reductions, then reconstruct final LayerNorm bias coordinates 0 and 3 as zero.

EVIDENCE: Fixing final bias coordinate 0 achieved 99.83% at 1495 parameters; additionally fixing coordinate 2 reached 98.03%, substantially closer to qualification than coordinate 1 at 71.26%, motivating the adjacent untested coordinate 3 as the next coordinate-sensitivity probe.

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


class FirstAndFourthBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinates 0 and 3 fixed at zero."""
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
                self.bias[:2],
                self.bias.new_zeros(1),
                self.bias[2:],
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
        self.ln_f = FirstAndFourthBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE