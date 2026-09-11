MECHANISM: Incremental pre-attention LayerNorm scale tying

HYPOTHESIS: Tying a third `ln1` scale coordinate will reduce the model from 1,579 to 1,578 parameters while retaining at least 99% accuracy, because the learned QKV weight columns can absorb the tied scale and the all-ones initialization remains functionally and RNG-identical.

INTENDED_EDIT: Store five rather than six learned `AnchoredLayerNorm` scales and reconstruct its final three scales by tying them to the first three learned coordinates.

EVIDENCE: The immediately preceding extension from one to two dynamically tied `ln1` scales achieved 99.92% accuracy at 1,579 parameters, while several constraints in other components failed; extending the same successful gauge by one coordinate is the closest controlled reduction.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and two dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:2]))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and three dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 3))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:3]))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE