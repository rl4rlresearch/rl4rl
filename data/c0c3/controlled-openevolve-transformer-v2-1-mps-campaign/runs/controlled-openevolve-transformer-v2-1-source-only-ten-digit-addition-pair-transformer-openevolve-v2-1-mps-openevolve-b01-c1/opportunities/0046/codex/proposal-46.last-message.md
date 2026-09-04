MECHANISM: LayerNorm-beta/downstream-bias gauge reorientation

HYPOTHESIS: Removing one `fc1` bias scalar will reduce the model from 1,579 to 1,578 parameters while retaining at least 99% accuracy, because `ln2`’s six learned bias coordinates can generate the omitted neuron offset through its learned weight row, while the other `fc1` biases cancel collateral offsets.

INTENDED_EDIT: Replace the MLP’s first linear layer with an otherwise identical layer whose final output-bias coordinate is fixed at zero, preserving all successful attention, embedding, final-LayerNorm, initialization, and optimizer settings.

EVIDENCE: Removing a third `ln2` bias coordinate collapsed accuracy to 40.61%, while the current design with six learned `ln2` bias coordinates reaches 100%; this motivates preserving that optimization pathway and instead removing one downstream bias coordinate from the same exact affine redundancy.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = MeanFreeResidualLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class OnePrunedOutputBiasLinear(nn.Linear):
    """Linear layer with one bias scalar absorbed by the preceding LayerNorm bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = OnePrunedOutputBiasLinear(d_model, d_ff)
        self.fc2 = MeanFreeResidualLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE