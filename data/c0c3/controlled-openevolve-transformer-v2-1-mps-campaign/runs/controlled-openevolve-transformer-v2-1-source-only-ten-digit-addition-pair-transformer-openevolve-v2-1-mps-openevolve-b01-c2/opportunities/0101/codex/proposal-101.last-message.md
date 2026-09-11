MECHANISM: Alternative sixth attention-LayerNorm scale quotient at coordinate 1

HYPOTHESIS: Fixing attention LayerNorm coordinate 1 and absorbing its scale into QKV weights will produce a 1,534-parameter model with at least 99% accuracy.

INTENDED_EDIT: Fix coordinate 1 alongside the five verified trailing LayerNorm scales, and expose the reconstructed dense scale vector to the sensitive-row QKV optimizer.

EVIDENCE: The 1,535-parameter model reached 99.82% with five fixed LayerNorm scales, while coordinate 1 also tolerated the latest successful positional quotient; testing it distinguishes coordinate-specific optimization sensitivity from coordinate 2’s 94.20% sixth-scale failure.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)
=======
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        # Coordinate 1 and the final five coordinates are fixed; their scales
        # are absorbed by the corresponding QKV input columns.
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def dense_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:],
                self.weight.new_ones(5),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, self.normalized_shape, self.dense_weight(), None, 1e-5
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scales = torch.cat(
            (
                normalization.weight,
                normalization.weight.new_ones(
                    module.in_features - normalization.weight.numel()
                ),
            )
        )
=======
        scales = normalization.dense_weight()
>>>>>>> REPLACE