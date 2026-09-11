MECHANISM: Scale-aligned pre-attention LayerNorm bias tying

HYPOTHESIS: Tying `ln1` bias coordinate 4 to learned bias coordinate 1 will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because it preserves an adaptable offset for coordinate 4 and follows the only successful partner mapping previously found for its LayerNorm scale.

INTENDED_EDIT: Store four learned `AnchoredLayerNorm` biases and reconstruct bias coordinate 4 from coordinate 1, while keeping coordinates 5–7 fixed at zero and preserving all verified scale mappings.

EVIDENCE: Fixing bias coordinate 4 at zero failed at 69.18%, showing that coordinate needs adaptability; independently, tying scale coordinate 4 to coordinate 1 achieved 99.73%, whereas tying it to coordinates 0, 2, or 3 failed, making bias coordinate 1 the best-supported partner.

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
    """LayerNorm with three anchored and one dynamically tied bias."""

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
        bias = torch.cat(
            (
                self.bias,
                self.bias[1:2],
                self.bias.new_zeros(3),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE