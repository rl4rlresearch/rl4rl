MECHANISM: Redundant first-quartet pre-attention bias anchoring

HYPOTHESIS: Fixing `ln1` bias coordinate 3 at zero will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because coordinates 0–2 remain independently adaptable and coordinate 4—whose removal or tying failed—remains learned.

INTENDED_EDIT: Store `ln1` biases for coordinates 0–2 and 4, reconstruct coordinate 3 and coordinates 5–7 as zero, and preserve all verified scale mappings and initialization behavior.

EVIDENCE: Fixing biases 5–7 achieved 99.92% at 1,576 parameters, whereas fixing or tying coordinate 4 failed; this isolates a different reduction while retaining the evidently essential independent coordinate-4 bias.

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
        bias = torch.cat(
            (
                self.bias[:3],
                self.bias.new_zeros(1),
                self.bias[3:],
                self.bias.new_zeros(3),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE