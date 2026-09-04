MECHANISM: One-coordinate MLP-output bias constraint

HYPOTHESIS: Fixing the final MLP-output bias coordinate at zero will retain at least 99% accuracy after 45,000 updates while reducing the verified model from 1,142 to 1,141 learned parameters.

INTENDED_EDIT: Learn seven of the eight MLP-output bias coordinates and reconstruct the final coordinate as zero during the forward pass, preserving the qualified transformer backbone and constructor RNG consumption.

EVIDENCE: The current 1,142-parameter model achieved 100% accuracy, while fixing a fifth token-embedding scalar failed at 14.31%. Two attention-output bias coordinates were removable with 100% accuracy, motivating a one-scalar probe of the separate, previously unconstrained MLP-output bias.

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
        # Fix the final additive output coordinate at zero. Constructing the
        # full Linear above preserves constructor RNG consumption.
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_bias = F.pad(self.fc2.bias, (0, 1))
        return self.drop(
            F.linear(F.gelu(self.fc1(x)), self.fc2.weight, output_bias)
        )
>>>>>>> REPLACE