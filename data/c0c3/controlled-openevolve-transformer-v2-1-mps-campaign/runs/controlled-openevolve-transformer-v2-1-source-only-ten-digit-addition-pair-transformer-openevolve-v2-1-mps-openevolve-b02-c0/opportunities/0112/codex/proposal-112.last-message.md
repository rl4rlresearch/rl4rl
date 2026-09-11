MECHANISM: Adjacent pre-attention LayerNorm bias absorption

HYPOTHESIS: Fixing `ln1` bias coordinate 4 at zero will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because the verified adjacent constraint on coordinate 5 preserved 99.92% accuracy and the following learned attention projections can adapt to the additional fixed offset.

INTENDED_EDIT: Store four learned `AnchoredLayerNorm` biases and reconstruct the final four bias coordinates as zero, while preserving all verified scale mappings and initialization behavior.

EVIDENCE: The immediately preceding reduction fixed `ln1` bias coordinate 5 and achieved 99.92% accuracy at 1,576 parameters; extending the same successful adjacent bias constraint is the closest controlled test after reductions involving scales, positional gauges, attention-output columns, and `ln2` scales failed.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with three anchored biases and four dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight,
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:2],
            )
        )
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with four anchored biases and four dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight,
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:2],
            )
        )
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE