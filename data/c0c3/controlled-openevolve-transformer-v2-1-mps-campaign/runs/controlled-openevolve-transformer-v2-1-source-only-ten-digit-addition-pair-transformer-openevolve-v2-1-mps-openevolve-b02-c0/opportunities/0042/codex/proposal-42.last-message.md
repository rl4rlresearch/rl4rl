MECHANISM: Orthonormal attention-output common-mode gauge elimination

HYPOTHESIS: Representing the attention output-projection bias in a seven-dimensional zero-mean orthonormal basis will reduce the model from 1,626 to 1,625 parameters while retaining at least 99% accuracy, because common-mode attention bias is removed by downstream LayerNorms and the orthonormal parameterization preserves initialization and avoids coordinate-anchor optimization distortion.

INTENDED_EDIT: Add an output projection whose eight-dimensional bias is reconstructed from seven orthonormal contrast coordinates, and use it for the causal self-attention projection.

EVIDENCE: Orthonormal elimination of a positional common-mode gauge achieved 99.82% at 1,627 parameters after shared and stop-gradient parameterizations of the same gauge failed badly; this motivates applying the successful independent orthonormal quotient to the previously fragile attention-projection common-mode gauge.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class AttentionGaugeLinear(nn.Linear):
    """Linear projection with zero-mean output bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        basis = self.weight.detach().new_zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias_basis @ self.bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = AttentionGaugeLinear(d_model)
>>>>>>> REPLACE