MECHANISM: Incremental final-output bias absorption

HYPOTHESIS: Removing a fourth final-LayerNorm bias direction will reduce the model from 1,583 to 1,582 parameters while retaining at least 99% accuracy, because the same token-row-mean output-bias channel supported 99.91% accuracy after the third direction was removed.

INTENDED_EDIT: Reduce the learned final-LayerNorm bias basis from five to four coordinates while preserving all attention, MLP, value-bias, initialization, and training settings.

EVIDENCE: The immediately preceding three-direction final-LayerNorm quotient achieved 99.91% accuracy with 1,583 parameters, improving on the two-direction quotient’s 99.62%; this makes a one-coordinate extension of that demonstrated mechanism the most informative next reduction.

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
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 4))
>>>>>>> REPLACE