MECHANISM: Single-direction pre-MLP bias gauge fixing

HYPOTHESIS: Reducing the pre-MLP LayerNorm bias from seven to six coefficients will produce a 1,280-parameter transformer with at least 99% accuracy, because the omitted constant offset is exactly absorbable by `fc1.bias`.

INTENDED_EDIT: Remove one learned zero-mean `ln2` bias direction while retaining six bias coefficients and the quotient-input MLP.

EVIDENCE: The 1,281-parameter quotient-input design achieved 99.81% accuracy, while the seven-direction `ln2` design achieved 99.96%. Complete `ln2` bias removal harmed optimization, so a one-direction reduction is the narrowest test of this redundancy.

<<<<<<< SEARCH
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm with a zero-mean learned bias."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 2:
            raise ValueError("normalized_shape must be at least two")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 1))

        basis = torch.zeros(normalized_shape, normalized_shape - 1)
        for col in range(normalized_shape - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
=======
class QuotientBiasLayerNorm(nn.Module):
    """Scale-fixed LayerNorm retaining six zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 3:
            raise ValueError("normalized_shape must be at least three")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 2))

        basis = torch.zeros(normalized_shape, normalized_shape - 2)
        for col in range(normalized_shape - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
>>>>>>> REPLACE