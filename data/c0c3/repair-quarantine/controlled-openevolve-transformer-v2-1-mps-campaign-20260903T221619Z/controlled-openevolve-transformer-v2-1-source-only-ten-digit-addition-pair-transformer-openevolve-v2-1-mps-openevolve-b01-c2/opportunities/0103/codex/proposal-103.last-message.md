MECHANISM: Alternative attention-LayerNorm scale quotient at coordinate 0

HYPOTHESIS: Fixing attention LayerNorm coordinate 0 while preserving coordinates 1 and 2 as learned scales will reduce the verified 1,534-parameter model to 1,533 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Replace the unsuccessful coordinate-2 sixth-scale choice with an untested coordinate-0 anchor, and expose the reconstructed dense scale vector to the existing dense QKV quotient optimizer.

EVIDENCE: The current Helmert query-row design achieved 99.89% at 1,534 parameters. Fixing coordinate 2 previously fell to 94.20%, so testing a different coordinate of the same exact QKV column-scaling quotient isolates whether that failure was coordinate-specific.

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
        # Coordinate 0 joins the five trailing fixed scales; coordinates 1
        # and 2 remain learned.
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def dense_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight.new_ones(1),
                self.weight,
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