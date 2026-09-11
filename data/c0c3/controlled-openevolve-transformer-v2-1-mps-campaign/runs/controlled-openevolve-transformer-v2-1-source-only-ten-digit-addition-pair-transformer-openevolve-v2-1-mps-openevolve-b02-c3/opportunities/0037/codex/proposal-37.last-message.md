MECHANISM: One-direction pre-MLP bias gauge fixing

HYPOTHESIS: Reducing the qualified 1,041-parameter model’s pre-MLP LayerNorm bias from two learned directions to one will produce a 1,040-parameter transformer with at least 99% accuracy, because the removed offset through `fc1` remains representable by `fc1.bias`.

INTENDED_EDIT: Retain one learned zero-mean pre-MLP LayerNorm bias coefficient instead of two while preserving the qualified architecture and training procedure.

EVIDENCE: The 1,041-parameter two-direction design achieved 99.92% accuracy, and every preceding single-direction pre-MLP bias reduction retained at least 99.95% until that step, making one further incremental reduction the most directly supported test.

<<<<<<< SEARCH
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining two zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 7:
            raise ValueError("normalized_shape must be at least seven")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 6))

        basis = torch.zeros(normalized_shape, normalized_shape - 6)
        for col in range(normalized_shape - 6):
=======
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining one zero-mean bias direction."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 8:
            raise ValueError("normalized_shape must be at least eight")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 7))

        basis = torch.zeros(normalized_shape, normalized_shape - 7)
        for col in range(normalized_shape - 7):
>>>>>>> REPLACE