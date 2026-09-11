MECHANISM: Value-preserving attention affine-bias quotient

HYPOTHESIS: Removing one attention output-projection bias coordinate while retaining all eight value biases will reduce the model from 1,547 to 1,546 parameters and maintain at least 99% accuracy, because the full-rank value-to-output projection lets the retained value bias represent the omitted constant output direction.

INTENDED_EDIT: Use a mean-free attention output projection with six learned bias coordinates instead of seven, leaving its weights and the complete query/value bias pathway unchanged.

EVIDENCE: Removing all value biases reached only 76.51%, and their balanced seven-parameter reconstruction reached 10.9%, showing that the value-bias optimization pathway is load-bearing despite the affine redundancy. The verified 1,547-parameter model restores that pathway and reaches 99.88%, motivating removal from the opposite side of the quotient.

<<<<<<< SEARCH
        self.proj = MeanFreeResidualLinear(d_model, d_model)
=======
        self.proj = OnePrunedMeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.bias_basis @ self.weight
        full_bias = self.bias @ self.bias_basis.transpose(0, 1)
        return F.linear(x, full_weight, full_bias)


class OnePrunedOutputBiasLinear(nn.Linear):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.bias_basis @ self.weight
        full_bias = self.bias @ self.bias_basis.transpose(0, 1)
        return F.linear(x, full_weight, full_bias)


class OnePrunedMeanFreeResidualLinear(MeanFreeResidualLinear):
    """Mean-free residual map with one bias direction supplied by value bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.bias_basis @ self.weight
        bias_coordinates = F.pad(self.bias, (0, 1))
        full_bias = bias_coordinates @ self.bias_basis.transpose(0, 1)
        return F.linear(x, full_weight, full_bias)


class OnePrunedOutputBiasLinear(nn.Linear):
>>>>>>> REPLACE