MECHANISM: Orthonormal LayerNorm-nullspace parameterization

HYPOTHESIS: Constraining every MLP input-weight row to the zero-mean subspace will reduce the model from 1,490 to 1,478 parameters while retaining at least 99% accuracy, because `ln2` outputs have exactly zero coordinate sum and therefore the removed component cannot affect the learned function.

INTENDED_EDIT: Store each 12-by-8 `fc1` weight as 12-by-7 coefficients in a fixed orthonormal basis of the zero-mean subspace, preserving the initialized function while removing 12 unobservable parameters.

EVIDENCE: Centering the harmonic readout along a LayerNorm-null direction reduced parameters to 1,490 while achieving 99.98% accuracy; the same exact invariance applies independently to every `fc1` row because its input comes directly from a non-affine LayerNorm.

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
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

        # Non-affine LayerNorm produces zero-mean inputs, so the all-ones
        # component of every fc1 row is unobservable. Build an orthonormal
        # basis for the complementary zero-mean subspace.
        basis = torch.eye(d_model)[:, : d_model - 1]
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.register_buffer("fc1_basis", basis.T, persistent=False)

    def gauge_fix_fc1(self) -> None:
        with torch.no_grad():
            weight = self.fc1.weight
            centered = weight - weight.mean(dim=1, keepdim=True)
            self.fc1.weight = nn.Parameter(
                (centered @ self.fc1_basis.T).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve the baseline RNG stream by initializing the original table.
=======
        self.apply(self._init_weights)

        # Compress only after full initialization, preserving the original RNG
        # stream and the initialized function on LayerNorm outputs.
        for block in self.blocks:
            block.mlp.gauge_fix_fc1()

        # Preserve the baseline RNG stream by initializing the original table.
>>>>>>> REPLACE