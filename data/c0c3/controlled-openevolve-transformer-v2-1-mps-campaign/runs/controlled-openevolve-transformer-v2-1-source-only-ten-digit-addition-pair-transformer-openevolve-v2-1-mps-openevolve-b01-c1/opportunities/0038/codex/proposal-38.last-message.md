MECHANISM: Amplified token-row-mean output-bias channel

HYPOTHESIS: Scaling the fixed common final-LayerNorm bias while removing a fourth learned bias direction will retain at least 99% accuracy with 1,582 parameters, because stronger common activation lets smaller token-row means absorb omitted output biases with less context-dependent interference.

INTENDED_EDIT: Reduce the final-LayerNorm bias basis from five to four learned coordinates and increase its fixed common component from a unit-norm vector to an all-ones vector.

EVIDENCE: The three-direction quotient achieved 99.91% at 1,583 parameters, while the otherwise identical four-direction quotient narrowly missed at 98.28%; this tests whether optimization through the demonstrated token-row-mean channel, rather than insufficient attention or MLP capacity, caused the gap.

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with three bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 3)
        for j in range(normalized_shape - 3):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 3))
=======
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with four bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 4)
        for j in range(normalized_shape - 4):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.ones(normalized_shape),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 4))
>>>>>>> REPLACE