MECHANISM: MLP activation-dependent output gauge quotient

HYPOTHESIS: Removing the LayerNorm-invisible common mode of the one-neuron MLP’s output weight will reduce the model from 1,375 to 1,374 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Emit seven learned MLP output coordinates and append a fixed-zero eighth coordinate, while retaining the independently learned hidden bias as a scalar.

EVIDENCE: The 1,375-parameter model achieved 99.99% accuracy at 50,000 steps. Unlike the failed positional quotient, this isolates a single redundant output-weight direction; the MLP branch’s activation-dependent common-mode output is erased by downstream LayerNorm.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x) + self.fc2.bias.mean()
        return self.drop(self.fc2(F.gelu(hidden)))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1)
        self.hidden_bias = nn.Parameter(torch.zeros(()))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x) + self.hidden_bias
        output = self.fc2(F.gelu(hidden))
        output = F.pad(output, (0, 1))
        return self.drop(output)
>>>>>>> REPLACE