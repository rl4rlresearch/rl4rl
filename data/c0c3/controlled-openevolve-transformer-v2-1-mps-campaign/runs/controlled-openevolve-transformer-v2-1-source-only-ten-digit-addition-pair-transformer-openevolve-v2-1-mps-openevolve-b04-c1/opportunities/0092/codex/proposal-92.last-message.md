MECHANISM: Full MLP input-shift quotient

HYPOTHESIS: Gauge-fixing all 12 `fc1` weight rows will reduce the model from 887 to 875 learned parameters while retaining at least 99% accuracy, because non-affine `ln2` produces zero-mean inputs, making each row’s common-mode weight exactly unobservable.

INTENDED_EDIT: Replace `fc1` with a row-wise Householder gauge-fixed linear layer, preserve the original conceptual initialization stream and bias, and optimize its coordinates with ambient AdamW.

EVIDENCE: Gauge-fixing every narrow-QKV row retained 99.92% accuracy in the current design; `fc1` has the same exact input-shift symmetry because it consumes non-affine LayerNorm output.

<<<<<<< SEARCH
class OneRowInputGaugeLinear(nn.Module):
=======
class AllRowInputGaugeLinear(nn.Module):
    """Linear layer quotienting each input-weight row's common mode."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(in_features - 1))
                for _ in range(out_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))

        inv_sqrt = in_features ** -0.5
        reflector = torch.full((in_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Consume the same constructor-time draws as the replaced Linear.
        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        conceptual_bias = torch.empty(out_features)
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(conceptual_bias, -bound, bound)
        self._set_weight(conceptual_weight)
        with torch.no_grad():
            self.bias.copy_(conceptual_bias)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = (
            2.0 * (value @ self.reflector) / self.reflector_norm_sq
        )
        return value - projection.unsqueeze(-1) * self.reflector

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(conceptual_weight)
            for coordinates, row in zip(self.coordinates, transformed):
                coordinates.copy_(row[1:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.bias.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = torch.stack(list(self.coordinates), dim=0)
        weight = self._householder(F.pad(coordinates, (1, 0)))
        return F.linear(x, weight, self.bias)


class OneRowInputGaugeLinear(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = AllRowInputGaugeLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
=======
        elif isinstance(module, AllRowInputGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        (
            coordinates,
            block.attn.proj.reflector,
            block.attn.proj.reflector_norm_sq,
        )
=======
    ] + [
        (
            coordinates,
            block.mlp.fc1.reflector,
            block.mlp.fc1.reflector_norm_sq,
        )
        for block in model.blocks
        for coordinates in block.mlp.fc1.coordinates
    ] + [
        (
            coordinates,
            block.attn.proj.reflector,
            block.attn.proj.reflector_norm_sq,
        )
>>>>>>> REPLACE