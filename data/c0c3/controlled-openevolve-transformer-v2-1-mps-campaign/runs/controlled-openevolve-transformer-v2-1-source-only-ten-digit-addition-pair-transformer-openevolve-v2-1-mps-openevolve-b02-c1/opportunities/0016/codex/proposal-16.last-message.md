MECHANISM: Orthonormal residual gauge coordinates

HYPOTHESIS: Representing the MLP output in an orthonormal seven-dimensional zero-mean basis will produce a 1,564-parameter model with at least 99% accuracy, because it removes only the LayerNorm-invisible common mode while preserving isotropic initialization and well-conditioned gradients.

INTENDED_EDIT: Make `fc2` produce seven coordinates, including its bias, then project them through a fixed orthonormal basis spanning the eight-dimensional zero-mean residual subspace.

EVIDENCE: The zero-padded seven-coordinate MLP projection reached only 93.33%, despite being function-class equivalent modulo LayerNorm; its induced coordinate metric has an eightfold weak direction. The current 1,576-parameter gauge-fixed model reached 99.82%, motivating the same exact 12-parameter reduction with an orthonormal rather than asymmetric gauge.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model - 1)

        basis = torch.zeros(d_model - 1, d_model)
        for i in range(d_model - 1):
            scale = math.sqrt((i + 1) * (i + 2))
            basis[i, : i + 1] = 1.0 / scale
            basis[i, i + 1] = -(i + 1) / scale
        self.register_buffer("resid_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = y @ self.resid_basis
        return self.drop(y)
>>>>>>> REPLACE