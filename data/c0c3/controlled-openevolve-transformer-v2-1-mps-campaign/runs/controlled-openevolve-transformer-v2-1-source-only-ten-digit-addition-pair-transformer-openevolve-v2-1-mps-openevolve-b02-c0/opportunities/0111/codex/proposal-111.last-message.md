MECHANISM: Adjacent pre-attention LayerNorm bias absorption

HYPOTHESIS: Fixing `ln1` bias coordinate 5 at zero will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because the following QKV and attention-output affine transformations can absorb this bias and initialization remains unchanged.

INTENDED_EDIT: Store five learned `AnchoredLayerNorm` biases and fix the final three biases at zero, while preserving all verified scale mappings.

EVIDENCE: The 1,577-parameter design achieves 99.73% with two adjacent `ln1` biases already fixed at zero; extending that established bias constraint is better isolated than the failed reductions involving an additional scale tie, positional gauge, or attention-output column.

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
>>>>>>> REPLACE