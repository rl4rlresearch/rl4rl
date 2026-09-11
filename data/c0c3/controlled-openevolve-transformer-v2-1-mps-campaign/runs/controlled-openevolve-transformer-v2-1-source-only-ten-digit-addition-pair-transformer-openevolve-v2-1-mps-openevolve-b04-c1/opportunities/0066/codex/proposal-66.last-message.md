MECHANISM: Single-row input-shift quotient for QKV

HYPOTHESIS: Quotienting one `qkv` weight row will reduce the model from 1430 to 1429 parameters while retaining at least 99% accuracy, because non-affine `ln1` produces zero-mean inputs and scalar ambient AdamW preserves the established optimization geometry.

INTENDED_EDIT: Replace the first `qkv` weight row with seven Householder zero-mean coordinates, preserve the conceptual initialization and random stream, and add those coordinates to the existing ambient optimizer.

EVIDENCE: The 1430-parameter model reached 99.49%, and scalar Householder quotients with ambient AdamW succeeded for every tested attention-projection column; isolating one exact `qkv` row symmetry avoids the unverified batched implementation used by the unsuccessful 36-row attempt.

<<<<<<< SEARCH
    def forward(self) -> torch.Tensor:
        bias = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, bias) / self.reflector_norm_sq
        return bias - self.reflector * projection


class OneColumnGaugeLinear(nn.Module):
=======
    def forward(self) -> torch.Tensor:
        bias = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, bias) / self.reflector_norm_sq
        return bias - self.reflector * projection


class OneRowInputGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_coordinates = nn.Parameter(torch.empty(in_features - 1))
        self.rest_weight = nn.Parameter(
            torch.empty(out_features - 1, in_features)
        )

        inv_sqrt = in_features ** -0.5
        reflector = torch.full((in_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Consume the same constructor-time draw as the replaced Linear.
        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        self._set_weight(conceptual_weight)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * torch.dot(self.reflector, value) / self.reflector_norm_sq
        return value - self.reflector * projection

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed_first = self._householder(conceptual_weight[0])
            self.first_coordinates.copy_(transformed_first[1:])
            self.rest_weight.copy_(conceptual_weight[1:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.rest_weight.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        first = self._householder(first)
        weight = torch.cat((first.unsqueeze(0), self.rest_weight), dim=0)
        return F.linear(x, weight)


class OneColumnGaugeLinear(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = OneRowInputGaugeLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OneColumnShiftGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, OneColumnShiftGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, OneRowInputGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.proj.eighth_coordinates,
        )
    ]
    gauge_ids = {id(parameter) for parameter, _, _ in gauge_entries}
=======
            block.attn.proj.eighth_coordinates,
        )
    ] + [
        (
            block.attn.qkv.first_coordinates,
            block.attn.qkv.reflector,
            block.attn.qkv.reflector_norm_sq,
        )
        for block in model.blocks
    ]
    gauge_ids = {id(parameter) for parameter, _, _ in gauge_entries}
>>>>>>> REPLACE