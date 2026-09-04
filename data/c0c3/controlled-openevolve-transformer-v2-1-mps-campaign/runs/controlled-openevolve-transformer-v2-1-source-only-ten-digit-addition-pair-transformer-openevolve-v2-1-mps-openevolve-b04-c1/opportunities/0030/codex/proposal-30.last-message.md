MECHANISM: Narrow multiplicative feature mixer

HYPOTHESIS: A seven-channel SwiGLU MLP will retain at least 99% accuracy while reducing the model from 1606 to 1584 parameters, because multiplicative gating can represent conditional feature interactions more efficiently than twelve independent GELU channels.

INTENDED_EDIT: Replace the 12-channel additive GELU MLP with two learned seven-channel projections whose gated product feeds the output projection, while preserving all verified attention, normalization, embedding, and MLP output-bias capacity.

EVIDENCE: Removing an MLP output-bias coordinate collapsed accuracy to 35.36%, showing that the MLP branch is load-bearing, while attention sharing and further query-bias reduction also failed. This motivates preserving those capacities and instead challenging the shared assumption that carry-relevant interactions require a wider additive GELU representation.

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
        gated_width = d_ff // 2 + 1
        self.gate = nn.Linear(d_model, gated_width)
        self.value = nn.Linear(d_model, gated_width)
        self.fc2 = nn.Linear(gated_width, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate(x)) * self.value(x)
        return self.drop(self.fc2(hidden))
>>>>>>> REPLACE