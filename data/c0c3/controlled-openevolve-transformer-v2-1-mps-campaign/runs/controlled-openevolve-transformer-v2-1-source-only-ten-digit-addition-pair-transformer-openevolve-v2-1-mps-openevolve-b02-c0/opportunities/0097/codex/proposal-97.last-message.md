MECHANISM: Incremental pre-attention LayerNorm scale absorption

HYPOTHESIS: Tying a second `ln1` scale coordinate will reduce the model from 1,580 to 1,579 parameters while retaining at least 99% accuracy, because the unconstrained QKV weight columns can absorb both tied scales and the all-ones initialization remains functionally and RNG-identical.

INTENDED_EDIT: Store six rather than seven learned `AnchoredLayerNorm` scales and reconstruct its final two scales by tying them to the first two learned coordinates.

EVIDENCE: The verified 1,580-parameter design already dynamically ties one `ln1` scale and achieves 99.89%; extending that successful pre-attention gauge by one coordinate tests an untried redundancy after additional `ln1` bias, `ln2` scale, embedding, attention, and `fc2` constraints failed.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
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
>>>>>>> REPLACE