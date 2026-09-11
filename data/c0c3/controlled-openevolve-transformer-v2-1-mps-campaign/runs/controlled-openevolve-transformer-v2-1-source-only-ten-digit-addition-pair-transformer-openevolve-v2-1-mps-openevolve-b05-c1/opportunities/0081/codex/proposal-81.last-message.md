MECHANISM: Orthonormal quotient chart for attention outputs

HYPOTHESIS: Replacing the failed oblique zero-row attention projection with an orthonormal mean-zero output basis will preserve at least 99% accuracy while reducing the model from 1,272 to 1,264 parameters.

INTENDED_EDIT: Parameterize the attention projection’s input-dependent output in the seven-dimensional mean-zero subspace using an orthonormal Helmert basis, while retaining its unrestricted eight-coordinate bias and existing bias sharing.

EVIDENCE: Removing the same eight common-mode parameters with a zero-padded output row reached only 74.64%, despite those outputs being LayerNorm-invisible. That chart makes one observable direction eight times less responsive; an orthonormal chart removes the same exact redundancy without this conditioning distortion.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
=======
class MeanZeroOutputLinear(nn.Module):
    """Linear map with an orthonormal chart of mean-zero output vectors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")
        self.linear = nn.Linear(in_features, out_features - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

        basis = torch.zeros(out_features - 1, out_features)
        for i in range(1, out_features):
            scale = math.sqrt(i * (i + 1))
            basis[i - 1, :i] = 1.0 / scale
            basis[i - 1, i] = -i / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) @ self.basis + self.bias


class FixedBiasLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE