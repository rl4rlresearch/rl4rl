MECHANISM: Fifth attention-output scalar-shift gauge

HYPOTHESIS: A 1,378-parameter model will retain at least 99% accuracy because the verified 1,379-parameter design achieved 99.67%, while gauge-fixing a fifth attention-output column applies the same exact pre-final-LayerNorm symmetry already verified for four columns.

INTENDED_EDIT: Extend `GaugeFixedAttentionProjection` from four to five gauge-fixed weight columns while preserving omitted-coordinate initialization and ambient AdamW optimization.

EVIDENCE: The current four-column attention-output gauge reached 99.67% at 1,379 parameters; extending the identical symmetry by one column is the smallest controlled reduction from that qualified design.

<<<<<<< SEARCH
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and four output-shift gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and five output-shift gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(5)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 5)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        self.weight_rest.copy_(raw_weight[:, 5:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 5:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE