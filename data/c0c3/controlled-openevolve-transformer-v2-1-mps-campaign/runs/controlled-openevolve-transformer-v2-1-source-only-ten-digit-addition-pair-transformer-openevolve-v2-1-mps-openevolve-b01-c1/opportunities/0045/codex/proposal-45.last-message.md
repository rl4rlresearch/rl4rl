MECHANISM: Activation-isolated token-row output-bias quotient

HYPOTHESIS: Centering the scaled final-LayerNorm activation before adding its fixed common component will let the explicit token-row means replace the last learned final-bias direction, producing a 1,578-parameter model with at least 99% accuracy.

INTENDED_EDIT: Remove the final LayerNorm’s last learned bias parameter and reserve its common activation direction exclusively for the decay-free token-row-mean output-bias channel.

EVIDENCE: The optimizer-aligned row means reached 100% accuracy with one learned final-bias coordinate at 1,579 parameters, but simply deleting that coordinate fell to 41.71%; activation centering removes the context-dependent common component that otherwise interferes with row means while preserving the original initialization.

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
    """Final LayerNorm with an isolated token-row-mean output-bias channel."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        self.register_parameter("bias", None)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            x, self.normalized_shape, self.weight, None, self.eps
        )
        normalized = normalized - normalized.mean(dim=-1, keepdim=True)
        return normalized + self.fixed_bias
>>>>>>> REPLACE