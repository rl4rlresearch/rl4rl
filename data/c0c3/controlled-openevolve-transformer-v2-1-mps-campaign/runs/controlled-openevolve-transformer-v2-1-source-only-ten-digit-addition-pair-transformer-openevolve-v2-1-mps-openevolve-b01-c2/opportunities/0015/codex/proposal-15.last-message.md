MECHANISM: Final-residual common-mode bias gauge fixing

HYPOTHESIS: Fixing one coordinate of the MLP output bias at zero will reduce the verified 1,610-parameter model to 1,609 parameters while retaining at least 99% accuracy, because any uniform component of that bias is erased by the final LayerNorm and the remaining seven coordinates still span all functionally relevant bias differences.

INTENDED_EDIT: Retain seven learned `fc2` bias coordinates, synthesize one fixed zero coordinate, and preserve the existing linear weight shape and initialization path.

EVIDENCE: The distributed LayerNorm anchoring design achieved 99.97% accuracy with 1,610 parameters, while removing a second normalization scale caused a large failure; this motivates removing one exact additive gauge from the final residual branch instead of further reducing normalization capacity. The earlier attention-output bias anchor reached 97.77%, so targeting the later MLP output bias avoids perturbing activations consumed by another learned sublayer.

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
        # A uniform output-bias component is removed by the final LayerNorm.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc2.bias, self.fc2.bias.new_zeros(1)))
        x = F.linear(F.gelu(self.fc1(x)), self.fc2.weight, bias)
        return self.drop(x)
>>>>>>> REPLACE