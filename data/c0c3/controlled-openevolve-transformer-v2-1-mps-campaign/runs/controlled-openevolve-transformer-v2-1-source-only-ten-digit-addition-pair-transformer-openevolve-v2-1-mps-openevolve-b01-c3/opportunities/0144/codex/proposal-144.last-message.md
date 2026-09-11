MECHANISM: Endpoint-anchored shared normalization contrast

HYPOTHESIS: The resulting 854-parameter transformer will maintain at least 99% accuracy because it preserves the essential unique-versus-shared scale contrast while allowing the six-coordinate group, rather than the singleton, to learn its magnitude.

INTENDED_EDIT: Replace final LayerNorm scales `[a,b,c,c,c,c,c,1]` with `[1,b,b,b,b,b,b,1]`, reducing three learned scales to one.

EVIDENCE: The 855-parameter `[a,b,b,b,b,b,b,1]` design achieved 99.76%; merging the shared sextet with the anchor reached 97.55%, while merging the singleton with the sextet collapsed, leaving this third pairwise class merge as the most informative untested 854-parameter design.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one shared scale quintet and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[-1:].expand(4), self.weight.new_ones(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one shared scale sextet and fixed endpoints."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight.new_ones(1),
                self.weight.expand(x.size(-1) - 2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE