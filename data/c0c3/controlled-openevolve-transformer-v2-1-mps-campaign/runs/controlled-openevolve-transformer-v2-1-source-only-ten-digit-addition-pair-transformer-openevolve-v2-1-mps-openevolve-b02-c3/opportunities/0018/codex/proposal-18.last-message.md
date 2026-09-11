MECHANISM: Third-direction pre-attention LayerNorm bias gauge fixing

HYPOTHESIS: Restricting the verified 1,457-parameter model’s pre-attention LayerNorm bias from six basis coefficients to five will yield 1,456 parameters while retaining at least 99% accuracy, because the removed constant attention-input shift can be absorbed by query bias, key-softmax invariance, and attention-output bias.

INTENDED_EDIT: Remove one additional learned pre-attention LayerNorm bias direction while preserving five learned zero-mean bias coefficients.

EVIDENCE: Consecutive one-direction reductions from 1,459 to 1,458 and then 1,457 parameters retained 99.94% and 99.95% accuracy; another single-direction reduction is the narrowest test of the same successful mechanism, while the failed complete bias removal argues against a larger step.

<<<<<<< SEARCH
class GaugeFixedLayerNorm(nn.Module):
    """LayerNorm retaining six learned constant-shift directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 3:
            raise ValueError("normalized_shape must be at least three")

        self.norm = nn.LayerNorm(normalized_shape, bias=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 2))

        basis = torch.zeros(normalized_shape, normalized_shape - 2)
        for col in range(normalized_shape - 2):
=======
class GaugeFixedLayerNorm(nn.Module):
    """LayerNorm retaining five learned constant-shift directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 4:
            raise ValueError("normalized_shape must be at least four")

        self.norm = nn.LayerNorm(normalized_shape, bias=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 3))

        basis = torch.zeros(normalized_shape, normalized_shape - 3)
        for col in range(normalized_shape - 3):
>>>>>>> REPLACE