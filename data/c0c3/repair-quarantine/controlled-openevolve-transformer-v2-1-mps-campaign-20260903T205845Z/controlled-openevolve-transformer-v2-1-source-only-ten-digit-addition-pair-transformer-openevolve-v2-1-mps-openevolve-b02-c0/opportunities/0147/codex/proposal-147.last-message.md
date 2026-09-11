MECHANISM: Incremental LayerNorm-to-attention affine gauge elimination

HYPOTHESIS: Anchoring the next `ln1` bias coordinate at zero will reduce the model from 1,576 to 1,575 parameters while preserving at least 99% accuracy, because its downstream affine effect can be absorbed by the learned attention projections.

INTENDED_EDIT: Reduce `AnchoredLayerNorm` from five to four learned bias coordinates and reconstruct the remaining four as zeros.

EVIDENCE: The current 1,576-parameter design achieved 99.92% accuracy after anchoring one `ln1` bias coordinate, supporting another conservative one-coordinate reduction using the same redundancy.

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