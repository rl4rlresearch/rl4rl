MECHANISM: Ambient-state quotient of an attention output-projection column

HYPOTHESIS: Initialization-preserving quotienting of the first attention output-projection column will reduce the model from 1593 to 1592 parameters while retaining at least 99% accuracy when its seven coordinates are optimized with eight-dimensional ambient AdamW.

INTENDED_EDIT: Replace one dense attention projection column with a Householder zero-mean parameterization, preserve the conceptual dense initialization and random-number stream, and include its coordinates in the ambient-state optimizer.

EVIDENCE: Ambient AdamW enabled every sequential initialization-preserving `fc2` column quotient through the final dense column to meet the threshold, with the latest 1593-parameter design reaching 100%; the attention output projection has the same final-LayerNorm shift symmetry.

<<<<<<< SEARCH
class ShiftGaugeEmbedding(nn.Module):
=======
class OneColumnShiftGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_coordinates = nn.Parameter(torch.empty(out_features - 1))
        self.rest_weight = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Consume the same constructor-time random draw as Linear so all
        # subsequent modules retain the verified initialization stream.
        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        self._set_weight(conceptual_weight)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * torch.dot(self.reflector, value) / self.reflector_norm_sq
        return value - self.reflector * projection

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed_first = self._householder(conceptual_weight[:, 0])
            self.first_coordinates.copy_(transformed_first[1:])
            self.rest_weight.copy_(conceptual_weight[:, 1:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.rest_weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        first = self._householder(first)
        weight = torch.cat((first.unsqueeze(1), self.rest_weight), dim=1)
        return F.linear(x, weight)


class ShiftGaugeEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model, bias=False)
=======
        self.proj = OneColumnShiftGaugeLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
        elif isinstance(module, OneColumnShiftGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_entries = [
        (
            coordinates,
            block.mlp.fc2.reflector,
            block.mlp.fc2.reflector_norm_sq,
        )
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
    ]
=======
    gauge_entries = [
        (
            coordinates,
            block.mlp.fc2.reflector,
            block.mlp.fc2.reflector_norm_sq,
        )
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
    ] + [
        (
            block.attn.proj.first_coordinates,
            block.attn.proj.reflector,
            block.attn.proj.reflector_norm_sq,
        )
        for block in model.blocks
    ]
>>>>>>> REPLACE