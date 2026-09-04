MECHANISM: Incremental decay-free final-output bias absorption

HYPOTHESIS: Removing a sixth final-LayerNorm bias direction will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because the zero-weight-decay token-row-mean channel achieved 99.96% accuracy after absorbing five bias directions.

INTENDED_EDIT: Reduce the final-LayerNorm learned bias basis from three to two coordinates while preserving the successful embedding optimizer group and all attention, MLP, initialization, and decoding behavior.

EVIDENCE: The immediately preceding five-direction quotient achieved 99.96% accuracy at 1,581 parameters, improving slightly on the four-direction quotient’s 99.95%; this makes one additional coordinate removal the most direct test of the demonstrated absorption mechanism.

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with five bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 5)
        for j in range(normalized_shape - 5):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 5))
=======
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with six bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 6)
        for j in range(normalized_shape - 6):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
>>>>>>> REPLACE