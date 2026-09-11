MECHANISM: LayerNorm-hyperplane anchoring of one MLP input weight

HYPOTHESIS: Anchoring one `fc1` weight coordinate at zero will reduce the model from 1,607 to 1,606 parameters while retaining at least 99% accuracy after 10,000 steps, because LayerNorm confines each MLP input to a seven-dimensional affine hyperplane and the retained `fc1` bias preserves the omitted direction’s constant component.

INTENDED_EDIT: Store 95 of the 96 `fc1` weights, reconstruct the final coordinate as zero during forward passes, and add 1,000 low-learning-rate refinement steps.

EVIDENCE: The current 1,607-parameter model achieved 99.89% after 9,000 steps. The failed attention-projection weight gauge motivates testing a distinct redundancy localized to one MLP neuron, whose LayerNorm-constrained input and retained bias preserve representational capacity without another gauge-aware optimizer.

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
        self.fc1.weight = nn.Parameter(torch.empty(d_ff * d_model - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = torch.cat(
            (self.fc1.weight, self.fc1.weight.new_zeros(1))
        ).view(self.fc1.out_features, self.fc1.in_features)
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=9000)
=======
    p.add_argument("--train-steps", type=int, default=10000)
>>>>>>> REPLACE