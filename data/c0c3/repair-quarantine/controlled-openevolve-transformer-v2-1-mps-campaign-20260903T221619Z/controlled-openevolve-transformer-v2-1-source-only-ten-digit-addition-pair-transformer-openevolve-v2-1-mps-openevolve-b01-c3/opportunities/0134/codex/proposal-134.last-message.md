MECHANISM: Final-pair normalization-scale sharing

HYPOTHESIS: The resulting 860-parameter transformer will maintain at least 99% accuracy because the current 861-parameter model achieved 99.99%, and tying one pair of final LayerNorm scales preserves the verified rank-six embedding, six-unit MLP, and five-gain attention configuration.

INTENDED_EDIT: Represent the first seven final LayerNorm scales as `[a,b,c,d,e,f,f]` with the eighth scale fixed at one, removing one learned parameter.

EVIDENCE: The verified 861-parameter design reached 99.99%, while reducing its attention gains to four parameters failed at 78.54%; this motivates an isolated one-parameter compression in the final normalization instead of further constraining the load-bearing attention gains.

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
    """Layer normalization with a tied final scale pair and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[-1:].expand(1), self.weight.new_ones(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE