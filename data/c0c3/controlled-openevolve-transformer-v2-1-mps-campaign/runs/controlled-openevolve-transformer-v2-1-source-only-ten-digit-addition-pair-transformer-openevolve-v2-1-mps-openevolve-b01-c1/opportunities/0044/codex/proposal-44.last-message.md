MECHANISM: Complete decay-free final-output bias absorption

HYPOTHESIS: Removing the eighth and final learned final-LayerNorm bias direction will produce a 1,578-parameter model with at least 99% accuracy, because the optimizer-aligned token-row-mean channel reached 100% accuracy after absorbing the first seven directions.

INTENDED_EDIT: Replace the one-coordinate quotient final LayerNorm with a fixed common bias only, eliminating its last learned bias parameter while preserving the decay-free explicit token-row means.

EVIDENCE: Reparameterizing token-row means as independent Adam coordinates allowed the seven-direction quotient to achieve 100% accuracy at 1,579 parameters; this directly motivates testing complete absorption of the remaining bias direction.

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with seven bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 7)
        for j in range(normalized_shape - 7):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.fixed_bias + self.bias_basis @ self.bias
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with all bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        del self.bias
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.fixed_bias, self.eps
        )
>>>>>>> REPLACE