MECHANISM: Consensus-gated seven-direction MLP

HYPOTHESIS: Replacing the seventh independent gate with a variance-preserving learned consensus of the other six gates will reduce the model from 845 to 837 parameters while retaining at least 99% accuracy, because all seven independent value and output directions remain available while only one gating hyperplane is shared.

INTENDED_EDIT: Challenge the old assumption that every observable MLP direction needs an independently parameterized gate; retain seven value/output directions but synthesize the seventh gate from six learned gates, while preserving the verified conceptual initialization stream.

EVIDENCE: The seven-unit GLU achieved 99.94%, while tying the MLP readout collapsed accuracy to 37.01%; this specifically identifies independent output directions as load-bearing, but provides no evidence that all seven gate projections must also be independent.

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
        conceptual_out_features: int = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conceptual_out_features = (
            out_features
            if conceptual_out_features is None
            else conceptual_out_features
        )
        if self.conceptual_out_features < out_features:
            raise ValueError(
                "conceptual_out_features must cover learned outputs"
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
        # The conceptual width can exceed the learned width so a structured
        # projection can preserve the established initialization stream.
        conceptual_weight = torch.empty(
            self.conceptual_out_features, in_features
        )
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        conceptual_bias = torch.empty(self.conceptual_out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(conceptual_bias, -bound, bound)
        self._set_weight(conceptual_weight)
        with torch.no_grad():
            self.bias.copy_(conceptual_bias[: self.out_features])
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
            self.conceptual_out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)
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
        if d_ff < 2:
            raise ValueError("consensus gating requires at least two units")
        self.d_ff = d_ff
        self.gate_width = d_ff - 1
        self.fc1 = AllRowInputGaugeLinear(
            d_model,
            d_ff + self.gate_width,
            conceptual_out_features=2 * d_ff,
        )
        self.fc2 = AllColumnOutputGaugeLinear(d_ff, d_model)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.fc1(x)
        value = features[..., : self.d_ff]
        learned_gate = features[..., self.d_ff :]
        consensus_gate = learned_gate.sum(
            dim=-1, keepdim=True
        ) / math.sqrt(self.gate_width)
        gate = torch.cat((learned_gate, consensus_gate), dim=-1)
        hidden = value * torch.sigmoid(gate)
        return self.drop(self.fc2(hidden) + self.output_bias())
>>>>>>> REPLACE