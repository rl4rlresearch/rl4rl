MECHANISM: Gauge-fixed MLP output bias

HYPOTHESIS: Fixing one functionally redundant component of the MLP output bias will reduce the model from 1,644 to 1,643 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Preserve the MLP width, weights, initialization draw sequence, and seven learned output-bias components while fixing the final bias component to zero during the functional linear operation.

EVIDENCE: The 1,644-parameter model achieved 99.96% accuracy, whereas larger reductions to 1,610 and 1,636 parameters failed; this motivates a one-parameter reduction along a uniform-shift gauge direction that the downstream LayerNorm removes.

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
        self.fc2.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.fc2.bias, (0, 1))
        return self.drop(F.linear(F.gelu(self.fc1(x)), self.fc2.weight, bias))
>>>>>>> REPLACE