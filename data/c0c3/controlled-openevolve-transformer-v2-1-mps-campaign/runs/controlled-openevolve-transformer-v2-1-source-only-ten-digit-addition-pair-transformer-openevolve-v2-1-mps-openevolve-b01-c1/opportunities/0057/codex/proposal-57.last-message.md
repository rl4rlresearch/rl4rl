MECHANISM: Reverse value-bias/output-bias affine quotient

HYPOTHESIS: Removing one attention output-projection bias coordinate will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because the full learned value bias can generate the omitted context-independent output offset through the learned projection.

INTENDED_EDIT: Prune the final mean-free bias coordinate only from the attention output projection, while preserving its full weight matrix, all value-bias coordinates, initialization draws, and every other model and training setting.

EVIDENCE: Removing one value-projection bias coordinate fell to 79.24% at 1,577 parameters, indicating that the value-bias optimization pathway is important; this patch tests the reverse orientation of the same exact affine redundancy by retaining the full value bias and removing a downstream projection-bias coordinate instead.

<<<<<<< SEARCH
        self.proj = MeanFreeResidualLinear(d_model, d_model)
=======
        self.proj = OnePrunedBiasMeanFreeResidualLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
class OnePrunedOutputBiasLinear(nn.Linear):
    """Linear layer with one bias scalar absorbed by the preceding LayerNorm bias."""
=======
class OnePrunedBiasMeanFreeResidualLinear(MeanFreeResidualLinear):
    """Mean-free residual map with one bias coordinate supplied by value bias."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.bias_basis @ self.weight
        full_bias = F.pad(self.bias, (0, 1)) @ self.bias_basis.transpose(0, 1)
        return F.linear(x, full_weight, full_bias)


class OnePrunedOutputBiasLinear(nn.Linear):
    """Linear layer with one bias scalar absorbed by the preceding LayerNorm bias."""
>>>>>>> REPLACE