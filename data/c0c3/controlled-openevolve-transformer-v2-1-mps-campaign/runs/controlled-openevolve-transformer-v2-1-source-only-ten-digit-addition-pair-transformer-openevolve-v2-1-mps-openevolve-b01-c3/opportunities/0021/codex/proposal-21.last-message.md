MECHANISM: Orthonormal LayerNorm input gauge fixing

HYPOTHESIS: Projecting the affine-free `ln2` output onto an orthonormal seven-dimensional mean-zero basis before the MLP will reduce the verified 1,527-parameter model to 1,515 parameters while retaining at least 99% accuracy, because LayerNorm removes the all-ones input component and the projection preserves every function available to `fc1`.

INTENDED_EDIT: Replace the MLP’s 8-to-12 first projection with a 7-to-12 projection applied to orthonormal coordinates of its mean-zero LayerNorm input, removing one exact null weight direction from each of 12 neurons.

EVIDENCE: The 1,527-parameter current design achieved 99.8% accuracy. The failed single-coordinate QKV gauge used an asymmetric coordinate anchor; this tests the same exact LayerNorm null-space principle in the MLP with an orthonormal, well-conditioned parameterization whose Gaussian effective-weight initialization matches the original projection onto the usable subspace.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.register_buffer(
            "input_basis", mean_zero_basis(d_model), persistent=False
        )
        self.fc2 = MeanZeroLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.input_basis
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE