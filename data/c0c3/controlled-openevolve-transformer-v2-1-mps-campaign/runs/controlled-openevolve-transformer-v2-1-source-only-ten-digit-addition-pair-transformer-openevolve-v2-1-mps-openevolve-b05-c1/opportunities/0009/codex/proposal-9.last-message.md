MECHANISM: Final-residual bias gauge anchoring

HYPOTHESIS: Fixing one coordinate of the MLP output bias at zero will reduce the model from 1,596 to 1,595 parameters while retaining at least 99% accuracy, because any removed bias value is equivalent to a uniform residual shift that the final LayerNorm cancels.

INTENDED_EDIT: Replace `fc2` with a linear layer that learns seven bias coordinates and fixes the eighth at zero; leave all weights, widths, initialization, training, and decoding unchanged.

EVIDENCE: The 1,596-parameter design achieved 99.99% accuracy, while jointly reparameterizing two residual biases fell to 13.33%; isolating the simplest single scalar gauge tests the same redundancy with minimal optimization disruption.

<<<<<<< SEARCH
class MLP(nn.Module):
=======
class AnchoredBiasLinear(nn.Linear):
    """Linear layer with the final bias coordinate fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        self.free_bias = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.free_bias, (0, 1))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = AnchoredBiasLinear(d_ff, d_model)
>>>>>>> REPLACE