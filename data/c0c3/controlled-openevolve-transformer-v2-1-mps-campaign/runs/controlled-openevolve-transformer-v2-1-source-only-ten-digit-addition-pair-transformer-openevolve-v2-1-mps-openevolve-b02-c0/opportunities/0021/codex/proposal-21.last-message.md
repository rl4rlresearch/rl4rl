MECHANISM: Joint LayerNorm shift-and-scale gauge anchoring

HYPOTHESIS: Fixing the last `ln1` scale coordinate at one will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because its matching bias coordinate is already fixed at zero and any learned scale can be absorbed exactly into the corresponding QKV weight column.

INTENDED_EDIT: Store seven learned `ln1` scale coordinates and append one fixed-one coordinate during the forward pass, preserving the initial function exactly.

EVIDENCE: Anchoring the last `ln1` bias coordinate previously achieved 99.95% accuracy; extending that successful parameterization at the same coordinate removes an exact LayerNorm-to-linear scale redundancy without altering initialization, unlike the failed final-MLP weight anchor.

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
    """LayerNorm with one fixed-one scale and fixed-zero bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE