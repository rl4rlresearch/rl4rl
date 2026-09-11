MECHANISM: Contrast-only final normalization scale

HYPOTHESIS: The resulting 854-parameter transformer will maintain at least 99% accuracy because it preserves the unique-versus-shared scale contrast lost by the failed septuple-sharing design, while fixing the shared sextet to the existing unit anchor.

INTENDED_EDIT: Replace the two learned final-LayerNorm scales `[a,b,b,b,b,b,b,1]` with one learned contrast scale `[a,1,1,1,1,1,1,1]`.

EVIDENCE: The 855-parameter sextuple-shared design achieved 99.76%, whereas tying its unique scale into the shared group collapsed accuracy to 10.28%; this suggests the unique coordinate contrast is essential, while the shared group’s absolute scale is the safer parameter to anchor.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one shared scale sextet and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[-1:].expand(5), self.weight.new_ones(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one learned contrast scale."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(x.size(-1) - 1)))
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE