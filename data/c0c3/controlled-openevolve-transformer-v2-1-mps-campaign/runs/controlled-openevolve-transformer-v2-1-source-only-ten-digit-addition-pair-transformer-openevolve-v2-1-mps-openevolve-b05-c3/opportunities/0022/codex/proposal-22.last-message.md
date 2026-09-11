MECHANISM: Single MLP hidden-bias ablation

HYPOTHESIS: Fixing one `fc1` bias coordinate at zero will produce 1600 learned parameters while retaining at least 99% accuracy, because all 12 nonlinear units and their input/output weights remain trainable, and the learned `ln2` affine offset can still shift the affected unit’s preactivation.

INTENDED_EDIT: Preserve the verified positional quotient and four-coordinate query bias, while representing `fc1.bias` with 11 learned coordinates and one fixed-zero trailing coordinate.

EVIDENCE: The 1601-parameter design reached 99.96%, while reductions in query bias, `ln2`, and `fc2` failed; this motivates preserving those load-bearing paths and testing the previously untouched, zero-initialized `fc1` bias with a localized one-scalar ablation.

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
        # Construct the baseline layer first to preserve its initialization RNG
        # stream, then fix one hidden-unit bias coordinate at zero.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE