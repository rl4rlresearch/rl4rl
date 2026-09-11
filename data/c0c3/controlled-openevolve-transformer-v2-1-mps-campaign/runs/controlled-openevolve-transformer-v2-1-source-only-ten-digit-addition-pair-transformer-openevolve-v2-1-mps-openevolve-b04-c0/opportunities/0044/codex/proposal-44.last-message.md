MECHANISM: Final-LayerNorm common-shift gauge fixing

HYPOTHESIS: Fixing coordinate 3 of `fc2.bias` at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the sole block is followed immediately by final LayerNorm, which makes any common shift across all MLP output coordinates unobservable.

INTENDED_EDIT: Replace the eight-parameter second MLP bias with seven learned coordinates and reconstruct coordinate 3 as a fixed zero during the forward pass.

EVIDENCE: The 1607-parameter design reached 99.91% accuracy, while the next attention-side bias constraint collapsed to 36.34%; this motivates testing a distinct exact null direction, using coordinate 3 because its pre-MLP gain and bias constraints both previously retained at least 99.9% accuracy.

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
        self.fc2.bias = None
        self.fc2_bias_rest = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.fc2_bias_rest.new_zeros(1)
        bias = torch.cat(
            (self.fc2_bias_rest[:3], zero, self.fc2_bias_rest[3:])
        )
        hidden = F.gelu(self.fc1(x))
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
>>>>>>> REPLACE