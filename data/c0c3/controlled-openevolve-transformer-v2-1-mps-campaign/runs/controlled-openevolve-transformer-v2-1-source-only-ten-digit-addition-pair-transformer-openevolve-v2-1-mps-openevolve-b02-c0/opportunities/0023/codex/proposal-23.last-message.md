MECHANISM: Dynamic LayerNorm scale tying

HYPOTHESIS: Tying the last `ln1` scale coordinate to the first will reduce the model from 1,634 to 1,633 parameters while retaining at least 99% accuracy, because the following QKV weight columns can absorb independent scale differences while the tied coordinate remains dynamically learned.

INTENDED_EDIT: Store seven `ln1` scale parameters and reuse the first learned scale as the eighth coordinate, preserving the initial all-ones scale exactly.

EVIDENCE: Fixing the last `ln1` scale at one failed at 37.17%, while dynamically tying the otherwise invariant key-bias scalar to a query-bias coordinate recovered 99.32% at 1,634 parameters; this directly motivates replacing a fixed scale anchor with a learned tie.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with one fixed-zero bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), self.weight, bias, 1e-5)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with anchored bias and dynamically tied scale coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE