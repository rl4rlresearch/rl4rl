MECHANISM: Reverse pre-MLP shift-to-bias gauge fixing

HYPOTHESIS: Fixing coordinate 0 of `fc1.bias` at zero will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because either remaining learned pre-MLP LayerNorm shift can absorb that scalar while the other hidden-unit biases compensate for its distributed effect.

INTENDED_EDIT: Replace the first MLP projection with an initialization-equivalent linear layer whose bias has eleven learned coordinates and a fixed-zero coordinate 0.

EVIDENCE: The 1607-parameter design achieved 99.91% accuracy while pre-MLP bias coordinates 2 and 5 remained learned; attempts to fix those shifts failed, so this patch preserves both sensitive coordinates and tests the reverse redundancy after six other pre-MLP shifts were successfully absorbed by `fc1.bias`.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class BiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinate 0 absorbed by its input shift."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((zero, self.bias_rest))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = BiasAnchoredLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE