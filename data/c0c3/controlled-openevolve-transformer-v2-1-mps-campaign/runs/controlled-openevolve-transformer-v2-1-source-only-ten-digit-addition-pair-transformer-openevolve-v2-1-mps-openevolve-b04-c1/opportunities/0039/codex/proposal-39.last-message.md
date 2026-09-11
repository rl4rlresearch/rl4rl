MECHANISM: Initialization-preserving third MLP column quotient

HYPOTHESIS: Reparameterizing a third `fc2` column while exactly preserving the successful 1603-parameter model’s initialization stream and initial function modulo final-LayerNorm shifts will achieve at least 99% accuracy with 1602 parameters.

INTENDED_EDIT: Split the first dense remainder column into a third Householder zero-mean coordinate row, initialize it from the same conceptual dense weight, and preserve all random draws used by the successful two-column design.

EVIDENCE: Two quotiented `fc2` columns achieved 99.98% accuracy, whereas the naive third quotient collapsed to 28.45%; because this is an exact final-residual symmetry, controlling the otherwise changed seeded initialization isolates the added quotient’s optimization effect.

<<<<<<< SEARCH
class OneColumnGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.rest = nn.Linear(in_features - 2, out_features, bias=False)
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
        nn.init.normal_(self.first_coordinates, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        projection = 2.0 * (first @ self.reflector) / self.reflector_norm_sq
        first = first - projection.unsqueeze(1) * self.reflector
        weight = torch.cat((first.transpose(0, 1), self.rest.weight), dim=1)
        return F.linear(x, weight)
=======
class OneColumnGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.rest_weight = nn.Parameter(torch.empty(out_features, in_features - 3))
        self.first_coordinates = nn.Parameter(torch.empty(3, out_features - 1))

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Match the random draws made by the verified two-column module.
        compatibility_weight = torch.empty(out_features, in_features - 2)
        nn.init.kaiming_uniform_(compatibility_weight, a=math.sqrt(5))
        with torch.no_grad():
            nn.init.normal_(self.first_coordinates[:2], mean=0.0, std=0.02)
            self.first_coordinates[2].zero_()
            self.rest_weight.zero_()

    def reset_reduced_parameters(self) -> None:
        with torch.no_grad():
            dense = torch.empty(
                self.out_features,
                self.in_features - 2,
                device=self.rest_weight.device,
                dtype=self.rest_weight.dtype,
            )
            nn.init.normal_(dense, mean=0.0, std=0.02)

            third = dense[:, 0] - dense[:, 0].mean()
            encoded = third - (
                2.0 * torch.dot(self.reflector, third) / self.reflector_norm_sq
            ) * self.reflector
            self.first_coordinates[2].copy_(encoded[1:])
            self.rest_weight.copy_(dense[:, 1:])

    @property
    def in_features(self) -> int:
        return self.rest_weight.shape[1] + 3

    @property
    def out_features(self) -> int:
        return self.rest_weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        projection = 2.0 * (first @ self.reflector) / self.reflector_norm_sq
        first = first - projection.unsqueeze(1) * self.reflector
        weight = torch.cat((first.transpose(0, 1), self.rest_weight), dim=1)
        return F.linear(x, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_reduced_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE