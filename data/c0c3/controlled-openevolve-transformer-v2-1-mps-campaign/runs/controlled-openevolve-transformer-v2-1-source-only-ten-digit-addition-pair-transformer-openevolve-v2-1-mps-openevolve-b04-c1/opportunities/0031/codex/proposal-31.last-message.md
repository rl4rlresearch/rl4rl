MECHANISM: Isotropic MLP output-bias gauge fixing

HYPOTHESIS: Reparameterizing the eight-coordinate MLP output bias as seven orthonormal zero-mean coordinates will reduce the model from 1606 to 1605 parameters while retaining at least 99% accuracy, because the discarded all-ones component is erased by the final LayerNorm.

INTENDED_EDIT: Replace the dense `fc2` bias with a seven-parameter Householder basis spanning the zero-mean subspace.

EVIDENCE: Anchoring one MLP bias coordinate failed at 35.36%, but the same isotropic quotient parameterization successfully removed the tied-embedding shift degree of freedom at 99.84%; this tests whether the prior failure came from the coordinate-asymmetric gauge rather than loss of functional capacity.

<<<<<<< SEARCH
class ShiftGaugeEmbedding(nn.Module):
=======
class ShiftGaugeBias(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.coordinates = nn.Parameter(torch.zeros(size - 1))

        inv_sqrt = size ** -0.5
        reflector = torch.full((size,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

    def forward(self) -> torch.Tensor:
        bias = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, bias) / self.reflector_norm_sq
        return bias - self.reflector * projection


class ShiftGaugeEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))) + self.output_bias())
>>>>>>> REPLACE