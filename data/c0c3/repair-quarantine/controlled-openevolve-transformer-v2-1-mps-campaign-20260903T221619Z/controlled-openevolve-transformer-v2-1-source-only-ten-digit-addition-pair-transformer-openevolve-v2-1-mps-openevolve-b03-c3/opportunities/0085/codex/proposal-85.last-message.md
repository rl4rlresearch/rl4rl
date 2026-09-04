MECHANISM: Tenth terminal-output scalar-shift gauge

HYPOTHESIS: A 1,382-parameter relative-lag model will retain at least 99% accuracy because the verified 1,383-parameter design achieved 99.74%, while removing the common output component of a tenth `fc2` column uses the identical pre-final-LayerNorm symmetry already verified for nine columns.

INTENDED_EDIT: Extend `GaugeFixedTerminalLinear` from nine to ten gauge-fixed weight columns, retaining the existing full ambient-gradient clipping and AdamW update path for the omitted coordinate.

EVIDENCE: The current nine-column-gauge design reached 99.74% at 1,383 parameters, and each prior extension from seven through nine columns remained above 99%, making the identical one-parameter reduction the most controlled next test.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and nine output-shift column gauges removed."""
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and ten output-shift column gauges removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(9)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 9)
        )
=======
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(10)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 10)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 9:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        self.weight_rest.copy_(raw_weight[:, 10:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 9:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 10:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE