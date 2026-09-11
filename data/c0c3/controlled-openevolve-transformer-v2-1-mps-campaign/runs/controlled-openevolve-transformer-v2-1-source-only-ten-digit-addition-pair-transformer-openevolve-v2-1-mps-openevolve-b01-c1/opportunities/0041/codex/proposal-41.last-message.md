MECHANISM: Incremental decay-free final-output bias absorption

HYPOTHESIS: Removing a fifth final-LayerNorm bias direction will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because exempting the tied token embedding from weight decay raised the four-direction quotient from 98.28% to 99.95%, demonstrating that its token-row means can reliably learn the replacement output-bias channel.

INTENDED_EDIT: Reduce the final-LayerNorm bias basis from four to three learned coordinates while preserving the successful zero-weight-decay embedding group and all attention, MLP, initialization, and training settings.

EVIDENCE: The immediately preceding four-direction quotient achieved 99.95% at 1,582 parameters only after token-embedding weight decay was removed; this strongly motivates testing one additional coordinate through the same demonstrated absorption mechanism.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE