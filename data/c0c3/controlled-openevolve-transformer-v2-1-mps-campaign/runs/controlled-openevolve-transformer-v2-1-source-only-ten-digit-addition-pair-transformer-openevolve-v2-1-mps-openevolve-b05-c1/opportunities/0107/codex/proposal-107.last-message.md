MECHANISM: Zero-overhead final-normalization scale tying

HYPOTHESIS: Tying the remaining coordinate-0 final-LayerNorm scale to the already learned coordinate-5 projection-null scale will produce a 1,265-parameter model with at least 99% accuracy while retaining baseline runtime and the exact freshly initialized function.

INTENDED_EDIT: Remove the dedicated final-LayerNorm scale parameter and use `shared_scales[0]` for both coordinates 0 and 5 without adding reductions or coefficient reconstruction.

EVIDENCE: The 1,266-parameter design achieved 100% accuracy, while several exact 1,265-parameter gauge reductions timed out after adding hot-path reconstruction. Both tied scales initialize to one, and projection-null scales have already trained successfully at 1,267 and 1,266 parameters.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with scales and bias stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with scales and bias stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)

    def forward(
        self,
        x: torch.Tensor,
        shared_scales: torch.Tensor,
        shared_bias: torch.Tensor,
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                shared_scales[:1],
                shared_scales.new_ones(1),
                shared_scales[1:2],
                shared_scales.new_ones(2),
                shared_scales[:1],
                shared_scales.new_ones(2),
            )
        )
>>>>>>> REPLACE