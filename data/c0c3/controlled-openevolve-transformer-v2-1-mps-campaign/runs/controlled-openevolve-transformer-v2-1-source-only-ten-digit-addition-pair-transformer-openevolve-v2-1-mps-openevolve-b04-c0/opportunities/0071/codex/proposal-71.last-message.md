MECHANISM: Alternative final-residual bias gauge anchoring

HYPOTHESIS: Fixing `fc2.bias[1]` instead of coordinate 0 will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy, because coordinate-specific optimization effects may make coordinate 1 removable even though fixing coordinate 0 failed.

INTENDED_EDIT: Replace the eight-parameter `fc2` bias with seven learned coordinates, reconstruct coordinate 1 as zero, and preserve ordinary `nn.Linear` initialization RNG consumption.

EVIDENCE: Fixing `fc2.bias[0]` reached only 2.79%, but the hidden-bias experiments showed that one symmetric coordinate can fail while another succeeds: fixing `fc1.bias[8]` reached 77.57%, whereas fixing coordinate 9 reached 100%.

<<<<<<< SEARCH
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
=======
        return F.linear(x, self.weight, bias)


class FinalBiasAnchoredLinear(nn.Linear):
    """Linear layer with bias coordinate 1 fixed at zero."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the ordinary bias first to preserve initialization RNG use.
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.bias_rest = nn.Parameter(torch.zeros(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = self.bias_rest.new_zeros(1)
        bias = torch.cat((self.bias_rest[:1], zero, self.bias_rest[1:]))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = FinalBiasAnchoredLinear(d_ff, d_model)
>>>>>>> REPLACE