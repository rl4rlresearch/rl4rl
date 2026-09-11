MECHANISM: Head-coordinate-aligned fifth pre-attention LayerNorm scale tie

HYPOTHESIS: Tying scale 3 to learned scale 1 will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because scale 1 was the only successful partner for the preceding tie and already controls corresponding scale 7 in the other attention head.

INTENDED_EDIT: Store three `AnchoredLayerNorm` scales and reconstruct scales 3–7 from learned scales 1, 1, 2, 0, and 1, preserving all four verified mappings.

EVIDENCE: The scale-4 tie to scale 1 achieved 99.73% at 1,577 parameters, while ties to scales 0, 2, and 3 failed; extending the uniquely successful partner to the remaining head-aligned coordinate is the closest controlled reduction.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and four dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight,
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:2],
            )
        )
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and five dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight,
                self.weight[1:2],
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:1],
                self.weight[1:2],
            )
        )
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE