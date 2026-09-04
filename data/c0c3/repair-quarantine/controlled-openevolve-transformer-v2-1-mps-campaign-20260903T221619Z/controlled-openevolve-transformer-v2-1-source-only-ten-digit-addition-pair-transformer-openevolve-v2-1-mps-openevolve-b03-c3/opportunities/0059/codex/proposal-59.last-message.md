MECHANISM: Seventh terminal-column output-shift gauge with ambient AdamW

HYPOTHESIS: A 1,531-parameter model will retain at least 99% accuracy because the verified 1,532-parameter design reached 99.91%, while a seventh `fc2` column’s common-output component is likewise erased by the final LayerNorm and its full AdamW dynamics remain preserved.

INTENDED_EDIT: Extend the terminal projection gauge from six to seven weight columns, retaining the existing ambient-coordinate gradient, clipping, moment, and update machinery.

EVIDENCE: Extending the terminal-column gauge from four to five columns achieved 99.89%, and extending it to six achieved 99.91%; applying the same exact symmetry to one additional column is the smallest controlled reduction beyond the current qualified design.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and six weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(6)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 6)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and seven weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(7)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 7)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 6:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 7:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 6:])
                nn.init.zeros_(module.bias)
=======
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 7:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and seven
    # terminal-weight gauges.
>>>>>>> REPLACE