MECHANISM: Balanced pre-attention LayerNorm bias quotient

HYPOTHESIS: Removing the common mode of the four currently learned `ln1` bias coordinates will reduce the model to 1,551 parameters while retaining at least 99% accuracy, because it preserves three relative directions spanning all four successful coordinates while downstream query/value biases absorb the omitted constant affine effect.

INTENDED_EDIT: Reparameterize the four learned `ln1` bias coordinates with a three-dimensional orthonormal mean-free basis instead of fixing an additional coordinate.

EVIDENCE: Four-coordinate pruning achieved 100% at 1,552 parameters, whereas directly fixing a fifth coordinate fell to 73.04%; preserving variation across all four previously learned coordinates tests whether that failure was caused by the coordinate-aligned parameterization.

<<<<<<< SEARCH
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm retaining mean-free variation across four bias coordinates."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(4, 3)
        for j in range(3):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias_basis @ self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE