MECHANISM: Circulant shared-feature GLU

HYPOTHESIS: Reusing one seven-feature learned basis for both values and gates, with a learned circulant gate transform, will retain at least 99% accuracy while reducing the model from 847 to 805 parameters.

INTENDED_EDIT: Replace the GLU’s fourteen independent input projections with seven shared projections, seven circulant mixing coefficients, and seven independent gate biases while preserving the established downstream initialization stream.

EVIDENCE: The verified seven-unit GLU reaches 99.94% accuracy at 847 parameters, showing that seven nonlinear output directions suffice; this tests the load-bearing assumption that every direction also requires a separate learned gate projection.

<<<<<<< SEARCH
class AllRowInputGaugeLinear(nn.Module):
    """Linear layer quotienting each input-weight row's common mode."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coordinates = nn.ParameterList(
=======
class AllRowInputGaugeLinear(nn.Module):
    """Linear layer quotienting each input-weight row's common mode."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initialization_out_features=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initialization_out_features = (
            out_features
            if initialization_out_features is None
            else initialization_out_features
        )
        if self.initialization_out_features < out_features:
            raise ValueError(
                "initialization_out_features must cover stored outputs"
            )
        self.coordinates = nn.ParameterList(
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Consume the same constructor-time draws as the replaced Linear.
        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        conceptual_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(conceptual_bias, -bound, bound)
        self._set_weight(conceptual_weight)
        with torch.no_grad():
            self.bias.copy_(conceptual_bias)
=======
        # Optionally consume a wider conceptual layer's draws so a shared
        # feature bank preserves the verified downstream initialization.
        conceptual_weight = torch.empty(
            self.initialization_out_features, in_features
        )
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        conceptual_bias = torch.empty(self.initialization_out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(conceptual_bias, -bound, bound)
        self._set_weight(conceptual_weight[:out_features])
        with torch.no_grad():
            self.bias.copy_(conceptual_bias[:out_features])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_parameters(self) -> None:
        conceptual_weight = self.bias.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)
        nn.init.zeros_(self.bias)
=======
    def reset_parameters(self) -> None:
        conceptual_weight = self.bias.new_empty(
            self.initialization_out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight[: self.out_features])
        nn.init.zeros_(self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = AllRowInputGaugeLinear(d_model, 2 * d_ff)
        self.fc2 = AllColumnOutputGaugeLinear(d_ff, d_model)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        hidden = value * torch.sigmoid(gate)
        return self.drop(self.fc2(hidden) + self.output_bias())
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_ff = d_ff
        self.fc1 = AllRowInputGaugeLinear(
            d_model,
            d_ff,
            initialization_out_features=2 * d_ff,
        )
        self.gate_mix = nn.Parameter(torch.zeros(d_ff))
        self.gate_bias = nn.Parameter(torch.zeros(d_ff))
        self.fc2 = AllColumnOutputGaugeLinear(d_ff, d_model)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.fc1(x)
        gate = features + self.gate_bias
        for offset, coefficient in enumerate(self.gate_mix):
            gate = gate + coefficient * torch.roll(
                features, shifts=offset, dims=-1
            )
        hidden = features * torch.sigmoid(gate)
        return self.drop(self.fc2(hidden) + self.output_bias())
>>>>>>> REPLACE