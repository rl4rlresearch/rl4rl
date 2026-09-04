MECHANISM: Three-pair final-normalization scale tying

HYPOTHESIS: Extending the verified 745-parameter final LayerNorm pattern from two disjoint tied pairs to three—`[a,b,b,c,c,d,d,1]`—will retain at least 99% accuracy while reducing the model to 744 learned parameters.

INTENDED_EDIT: Replace five learned final-normalization scales with four, tying the second and third coordinates while preserving the two already verified pairs and fixed anchor.

EVIDENCE: The current two-pair LayerNorm design achieved 99.89% accuracy with 745 parameters, and the preceding one-pair design achieved 99.8% with 746; unlike failed MLP-width and attention-gain reductions, successive normalization-scale tying has retained substantial accuracy margin.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with two paired scales and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:3],
                self.weight[3:4].expand(2),
                self.weight[4:5].expand(2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with three paired scales and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight[1:2].expand(2),
                self.weight[2:3].expand(2),
                self.weight[3:4].expand(2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE