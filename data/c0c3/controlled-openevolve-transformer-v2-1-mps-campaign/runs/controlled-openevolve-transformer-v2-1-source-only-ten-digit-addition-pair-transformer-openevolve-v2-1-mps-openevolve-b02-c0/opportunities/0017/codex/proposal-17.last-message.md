MECHANISM: Final-normalization common-mode bias anchoring

HYPOTHESIS: Fixing one coordinate of the final MLP output bias at zero will reduce the model from 1,636 to 1,635 parameters while retaining at least 99% accuracy, because the omitted common-mode bias degree of freedom is exactly removed by the following final LayerNorm.

INTENDED_EDIT: Replace `fc2` with an initialization-preserving linear layer that learns seven bias coordinates and appends one fixed-zero coordinate.

EVIDENCE: The current head-shared key-bias design achieved 99.92% at 1,636 parameters, while removing its last key-bias scalar collapsed accuracy; this motivates preserving attention unchanged and instead fixing an exact downstream bias gauge immediately before final normalization.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class OutputAnchoredLinear(nn.Linear):
    """Linear projection with the common-mode output bias gauge fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, F.pad(self.bias, (0, 1)))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = OutputAnchoredLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE