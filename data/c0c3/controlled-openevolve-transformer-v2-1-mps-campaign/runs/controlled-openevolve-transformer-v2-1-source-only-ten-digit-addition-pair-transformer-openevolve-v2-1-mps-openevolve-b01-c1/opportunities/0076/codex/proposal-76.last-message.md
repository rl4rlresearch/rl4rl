MECHANISM: Complete final-LayerNorm bias quotient

HYPOTHESIS: Fixing the remaining learned final-LayerNorm bias coordinate at its initialized zero will reduce the model from 1,547 to 1,546 parameters while retaining at least 99% accuracy, because the fixed common offset preserves the token-row-mean pathway that already supports eliminating the other seven bias parameters.

INTENDED_EDIT: Replace the final LayerNorm’s single learned bias coordinate with only its existing fixed common bias, leaving all attention, scale, embedding, and training pathways unchanged.

EVIDENCE: The verified 1,547-parameter model reaches 99.88% accuracy with seven of eight final-LayerNorm bias parameters already removed; extending that successful quotient is more targeted than modifying the attention-bias or LayerNorm-scale pathways whose recent one-parameter reductions failed.

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
    """Final LayerNorm with its bias absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        self.bias = None
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