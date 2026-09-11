MECHANISM: Isolated LayerNorm-nullspace compression

HYPOTHESIS: Compressing only the MLP-input map onto the seven-dimensional mean-zero LayerNorm subspace will reduce parameters from 1,596 to 1,584 while retaining at least 99% accuracy.

INTENDED_EDIT: Add an orthonormal mean-zero input projection and use it only for `fc1`, preserving its bias, output width, initialization scale, and full function class on normalized inputs.

EVIDENCE: Jointly compressing QKV and `fc1` reached 97.76%, close to the threshold, while the unchanged 1,596-parameter model reached 99.99%; isolating the smaller 12-parameter `fc1` reduction tests the exact redundancy with less optimization disruption.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MeanZeroInputLinear(nn.Module):
    """Linear map parameterized on the mean-zero input subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        spanning = torch.eye(in_features)[:, :-1]
        spanning[-1] = -1.0
        basis = torch.linalg.qr(spanning, mode="reduced").Q
        self.register_buffer("basis", basis, persistent=False)
        self.linear = nn.Linear(in_features - 1, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x @ self.basis)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE