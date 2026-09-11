MECHANISM: Rank-complete sigmoid-gated MLP bottleneck

HYPOTHESIS: Replacing the twelve-unit GELU MLP with a seven-unit learned GLU will reduce the model from 866 to 847 parameters while retaining at least 99% accuracy, because seven independent output directions span the entire LayerNorm-visible residual quotient and multiplicative gates provide greater nonlinear efficiency per direction.

INTENDED_EDIT: Add a generic gauge-fixed output projection, replace the GELU MLP with seven learned value/gate pairs, update ambient optimization for its columns, and set the default gated width to seven.

EVIDENCE: Tying the MLP readout collapsed accuracy to 37.01%, showing that independent output directions are load-bearing; this design preserves seven independent directions—the maximum observable rank after common-mode quotienting—while challenging the assumption that twelve separate GELU features are required.

<<<<<<< SEARCH
        return F.linear(x, weight)


class OneColumnShiftGaugeLinear(nn.Module):
=======
        return F.linear(x, weight)


class AllColumnOutputGaugeLinear(nn.Module):
    """Gauge-fixed projection spanning the observable residual quotient."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(in_features)
            ]
        )

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        self._set_weight(conceptual_weight)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = (
            2.0 * (value @ self.reflector) / self.reflector_norm_sq
        )
        return value - projection.unsqueeze(-1) * self.reflector

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(
                conceptual_weight.transpose(0, 1)
            )
            for coordinates, column in zip(
                self.coordinates, transformed
            ):
                coordinates.copy_(column[1:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.coordinates[0].new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = torch.stack(list(self.coordinates), dim=0)
        columns = self._householder(F.pad(coordinates, (1, 0)))
        return F.linear(x, columns.transpose(0, 1))


class OneColumnShiftGaugeLinear(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = AllRowInputGaugeLinear(d_model, d_ff)
        self.fc2 = OneColumnGaugeLinear(d_ff, d_model)
        self.output_bias = ShiftGaugeBias(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))) + self.output_bias())
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
        elif isinstance(module, OneColumnShiftGaugeLinear):
=======
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
        elif isinstance(module, AllColumnOutputGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, OneColumnShiftGaugeLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in model.blocks
        for coordinates in (
            block.mlp.fc2.third_coordinates,
            block.mlp.fc2.fourth_coordinates,
            block.mlp.fc2.fifth_coordinates,
            block.mlp.fc2.sixth_coordinates,
            block.mlp.fc2.seventh_coordinates,
            block.mlp.fc2.eighth_coordinates,
            block.mlp.fc2.ninth_coordinates,
            block.mlp.fc2.tenth_coordinates,
            block.mlp.fc2.middle_coordinates,
        )
=======
        for block in model.blocks
        for coordinates in block.mlp.fc2.coordinates
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE