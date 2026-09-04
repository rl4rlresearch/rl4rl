MECHANISM: Final-LayerNorm common-shift gauge removal

HYPOTHESIS: Removing one redundant scalar from the MLP output bias will retain at least 99% accuracy while reducing parameters from 1,644 to 1,643.

INTENDED_EDIT: Represent the final MLP bias with seven learned values and a fixed zero eighth value, preserving initialization RNG and full function capacity modulo the common shift erased by the final LayerNorm.

EVIDENCE: The 1,644-parameter baseline achieved 99.96%, while larger eight-parameter and width reductions failed; this motivates the smallest possible reduction in an exactly redundant direction with baseline constructor RNG preserved.

<<<<<<< SEARCH
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
=======
        y = self.resid_drop(y)
        return y


class LastZeroBiasLinear(nn.Linear):
    """Linear layer with the final, LayerNorm-redundant bias coordinate fixed."""

    def __init__(self, in_features: int, out_features: int):
        # Construct the baseline layer first to preserve its initialization RNG stream.
        super().__init__(in_features, out_features, bias=True)
        self.bias = nn.Parameter(self.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.linear(x, self.weight, bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = LastZeroBiasLinear(d_ff, d_model)
>>>>>>> REPLACE