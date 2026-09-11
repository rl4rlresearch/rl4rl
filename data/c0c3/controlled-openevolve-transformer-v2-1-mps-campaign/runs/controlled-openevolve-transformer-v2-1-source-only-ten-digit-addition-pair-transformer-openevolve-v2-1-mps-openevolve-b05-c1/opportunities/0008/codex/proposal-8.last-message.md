MECHANISM: Mean-zero residual-bias gauge fixing

HYPOTHESIS: Constraining the attention-output and MLP-output biases to their seven-dimensional mean-zero subspaces will reduce the model from 1,596 to 1,594 parameters while retaining at least 99% accuracy, because their discarded all-ones components are removed by subsequent LayerNorm operations.

INTENDED_EDIT: Add an orthonormally parameterized mean-zero-bias linear layer and use it for the attention projection and MLP output projection.

EVIDENCE: The 1,596-parameter design achieved 99.99% accuracy; unlike the failed 36-parameter LayerNorm-nullspace reduction, this patch removes only two independently unobservable scalar bias directions while preserving initialization and all output dimensions.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class MeanZeroBiasLinear(nn.Module):
    """Linear map whose learned bias spans the mean-zero output subspace."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias_coords = nn.Parameter(torch.zeros(out_features - 1))

        basis = torch.eye(out_features)[:, :-1] - torch.eye(out_features)[:, -1:]
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        self.register_buffer("bias_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight, self.bias_basis @ self.bias_coords)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = MeanZeroBiasLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = MeanZeroBiasLinear(d_ff, d_model)
>>>>>>> REPLACE