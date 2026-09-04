MECHANISM: Pair-shared final normalization scale

HYPOTHESIS: The resulting 859-parameter transformer will maintain at least 99% accuracy because the 860-parameter model achieved 99.86%, and sharing two adjacent final LayerNorm scales removes one parameter without further constraining the embedding codebook whose second anchored coefficient caused the prior 859-parameter model to collapse to 64.24%.

INTENDED_EDIT: Tie the final two trainable LayerNorm scale coordinates while retaining the existing fixed scale anchor, rank-six embedding, five-gain attention pattern, and six-unit MLP.

EVIDENCE: The single-coefficient-anchored 860-parameter design met the requirement at 99.86%, whereas anchoring a second embedding coefficient failed severely; this motivates testing the same one-parameter reduction in a different, localized component.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with seven learned scales and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one shared scale pair and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[-1:], self.weight.new_ones(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE