MECHANISM: Fixed global LayerNorm temperature quotient

HYPOTHESIS: Fixing the shared final LayerNorm scale at one will reduce the qualified 1,119-parameter model to 1,118 parameters while retaining at least 99% accuracy, because the scalar only controls logit temperature at the output and its internal effect can be represented by the learned value map.

INTENDED_EDIT: Replace the learned scalar final-LayerNorm scale with a fixed unit buffer while preserving the existing normalization and scale-sharing interfaces.

EVIDENCE: The current design achieved 99.99% accuracy at 1,119 parameters, while both attempted 1,118-parameter extensions of the constrained second-head projection collapsed; this tests an orthogonal redundancy without further restricting attention features or the token codebook.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and fully tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with fixed unit scale and zero bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model))

    def full_weight(self) -> torch.Tensor:
        return self.fixed_weight
>>>>>>> REPLACE