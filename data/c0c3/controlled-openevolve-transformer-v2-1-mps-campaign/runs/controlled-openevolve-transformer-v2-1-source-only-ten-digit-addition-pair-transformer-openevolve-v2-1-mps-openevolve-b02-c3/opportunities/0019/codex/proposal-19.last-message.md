MECHANISM: Fourth-direction pre-attention LayerNorm bias reduction

HYPOTHESIS: Restricting the verified 1,456-parameter model’s pre-attention LayerNorm bias from five learned directions to four will produce a 1,455-parameter transformer while retaining at least 99% accuracy.

INTENDED_EDIT: Parameterize the pre-attention LayerNorm bias with four basis coefficients instead of the current seven, advancing one parameter beyond the qualified five-coefficient design.

EVIDENCE: Successive reductions to seven, six, and five pre-attention bias coefficients achieved 99.94%, 99.95%, and 99.91% accuracy respectively; reducing one additional direction is the narrowest continuation of this demonstrated trend.

<<<<<<< SEARCH
class QuotientLayerNorm(nn.Module):
    """LayerNorm whose bias omits its attention-redundant all-ones direction."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, bias=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 1))

        basis = torch.zeros(normalized_shape, normalized_shape - 1)
        for col in range(normalized_shape - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
=======
class QuotientLayerNorm(nn.Module):
    """LayerNorm retaining four learned pre-attention bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 5:
            raise ValueError("normalized_shape must be at least five")

        self.norm = nn.LayerNorm(normalized_shape, bias=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 4))

        basis = torch.zeros(normalized_shape, normalized_shape - 4)
        for col in range(normalized_shape - 4):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
>>>>>>> REPLACE