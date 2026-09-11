MECHANISM: Third attention-output column gauge with ambient AdamW

HYPOTHESIS: A 1,530-parameter model will retain at least 99% accuracy because the verified two-column attention-projection gauge achieved 99.96%, while a third column’s common-output component is erased by downstream LayerNorm and the existing ambient-coordinate optimizer preserves its full AdamW dynamics.

INTENDED_EDIT: Extend the attention output-projection gauge from two to three weight columns, retaining the existing initialization, gradient clipping, moment, and update machinery.

EVIDENCE: The current 1,531-parameter design successfully reduced the attention projection from one gauged column to two at 99.96% accuracy; this tests the smallest possible extension along that successful reduction axis after the seventh terminal-column gauge failed.

<<<<<<< SEARCH
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and two weight-column output gauges removed."""

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
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and three weight-column output gauges removed."""

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
    # positional, two attention-weight, attention-bias, terminal-bias, and six
=======
    # positional, three attention-weight, attention-bias, terminal-bias, and six
>>>>>>> REPLACE