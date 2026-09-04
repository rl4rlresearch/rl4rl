MECHANISM: Three-column terminal MLP output-direction gauge fixing

HYPOTHESIS: Extending the qualified terminal gauge from two to three `fc2` weight columns will reduce the model to 1,623 parameters while retaining at least 99% accuracy, because the additional omitted all-ones output component is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.

INTENDED_EDIT: Gauge-fix the first three terminal MLP weight columns and update initialization/storage offsets accordingly; the existing ambient optimizer automatically includes the added gauge vector.

EVIDENCE: The one-column gauge achieved 99.99% accuracy at 1,625 parameters and the two-column extension achieved 99.96% at 1,624, directly motivating the smallest adjacent one-parameter extension.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and two weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(2)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 2)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and three weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(3)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 2:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 2:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for every stored
    # seven-coordinate gauge vector.
=======
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and three terminal-weight gauge vectors.
>>>>>>> REPLACE