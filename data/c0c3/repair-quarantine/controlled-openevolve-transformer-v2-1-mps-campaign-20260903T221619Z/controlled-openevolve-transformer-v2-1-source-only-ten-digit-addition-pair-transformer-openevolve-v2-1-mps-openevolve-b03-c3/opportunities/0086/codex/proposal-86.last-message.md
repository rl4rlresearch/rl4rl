MECHANISM: Eleventh terminal-output scalar-shift gauge

HYPOTHESIS: A 1,381-parameter relative-lag model will retain at least 99% accuracy because the verified 1,382-parameter ten-column design achieved 99.45%, while gauge-fixing an eleventh `fc2` column applies the same exact pre-final-LayerNorm output-shift symmetry.

INTENDED_EDIT: Reproduce the qualified ten-column terminal gauge and extend it to eleven columns, preserving the omitted coordinate through the existing ambient-gradient clipping and AdamW update path.

EVIDENCE: Reference Design 1 reached 99.45% accuracy at 1,382 parameters with ten terminal-column gauges; every tested extension through the tenth column remained above 99%, motivating the smallest controlled one-parameter reduction.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and eight weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(8)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 8)
        )
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and eleven output-shift column gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(11)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 11)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 8:])
=======
        self.weight_rest.copy_(raw_weight[:, 11:])
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 8:])
=======
                module.weight_rest.copy_(raw_weight[:, 11:])
>>>>>>> REPLACE