MECHANISM: Incremental pre-attention LayerNorm bias absorption

HYPOTHESIS: Fixing a second `ln1` bias coordinate to zero will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because query shifts can be absorbed by the learned query bias, key shifts cancel in attention softmax, and value shifts can be absorbed by the attention projection bias up to its irrelevant common mode.

INTENDED_EDIT: Store six rather than seven learned `AnchoredLayerNorm` bias coordinates and reconstruct its final two coordinates as zeros.

EVIDENCE: Removing every `ln2` bias coordinate retained 99.96% accuracy, and the current 1,582-parameter design already fixes one `ln1` bias coordinate while achieving 99.91%; extending this affine redundancy by one coordinate is a controlled alternative to the positional, `fc2`, and attention-weight gauges whose next adjacent constraints failed.

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
    """LayerNorm with two anchored biases and dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE