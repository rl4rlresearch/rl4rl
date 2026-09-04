MECHANISM: Second-direction pre-MLP bias gauge fixing

HYPOTHESIS: Reducing the qualified 1,280-parameter model’s pre-MLP LayerNorm bias from six learned directions to five will produce a 1,279-parameter transformer with at least 99% accuracy, because the removed constant offset remains absorbable by `fc1.bias`.

INTENDED_EDIT: Apply the verified quotient-input MLP and retain five learned zero-mean pre-MLP bias coefficients.

EVIDENCE: The six-coefficient quotient-input design achieved 99.95% with 1,280 parameters; the analogous pre-attention LayerNorm retained 99.91% with five coefficients, making a one-direction reduction the narrowest supported next test.

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
    """Scale-fixed LayerNorm retaining five zero-mean bias directions."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        if normalized_shape < 4:
            raise ValueError("normalized_shape must be at least four")

        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 3))

        basis = torch.zeros(normalized_shape, normalized_shape - 3)
        for col in range(normalized_shape - 3):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
=======
        self.fc1 = QuotientInputLinear(d_model, d_ff)
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
>>>>>>> REPLACE