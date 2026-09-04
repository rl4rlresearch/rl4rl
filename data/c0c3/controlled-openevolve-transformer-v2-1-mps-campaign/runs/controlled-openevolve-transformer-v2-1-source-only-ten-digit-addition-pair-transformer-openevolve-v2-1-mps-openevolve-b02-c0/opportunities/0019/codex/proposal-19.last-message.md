MECHANISM: Final-MLP weight common-mode gauge anchoring

HYPOTHESIS: Fixing one coordinate of the final MLP output weight at zero will reduce the model from 1,635 to 1,634 parameters while retaining at least 99% accuracy, because its contribution differs from an unrestricted weight only by a token-dependent common-mode offset removed by the final LayerNorm.

INTENDED_EDIT: Store 95 learned `fc2` weight coordinates, append one fixed-zero coordinate during the forward pass, and preserve the existing seven-coordinate anchored bias.

EVIDENCE: Anchoring the final MLP output bias achieved 99.95% accuracy at 1,635 parameters, while applying a similar anchor inside the attention projection failed; this motivates extending the successful final-MLP gauge by the smallest possible one-coordinate reduction.

<<<<<<< SEARCH
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with the common-mode output bias gauge fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, F.pad(self.bias, (0, 1)))
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with output-bias and one output-weight gauge fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1)).view(self.out_features, self.in_features)
        return F.linear(x, weight, F.pad(self.bias, (0, 1)))
>>>>>>> REPLACE