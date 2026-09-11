MECHANISM: Incremental dynamic LayerNorm scale tying

HYPOTHESIS: Tying the second-to-last `ln1` scale coordinate to the second learned coordinate will reduce the model from 1,633 to 1,632 parameters while retaining at least 99% accuracy, because the following QKV weight columns can absorb independent scale differences while all tied coordinates remain dynamically learned.

INTENDED_EDIT: Store six learned `ln1` scale parameters, reconstruct the seventh from the second and the eighth from the first, and preserve the initial all-ones scale exactly.

EVIDENCE: Dynamically tying the eighth `ln1` scale to the first achieved 99.78% accuracy at 1,633 parameters, whereas fixing it at one achieved only 37.17%; this supports one further incremental learned tie rather than a fixed anchor.

<<<<<<< SEARCH
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
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with anchored bias and dynamically tied scale coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[1:2], self.weight[:1]))
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE