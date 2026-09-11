MECHANISM: Initialization-preserving last-channel MLP quotient

HYPOTHESIS: Quotienting the last `fc2` input column instead of the optimization-sensitive third column will reduce the model from 1603 to 1602 parameters while retaining at least 99% accuracy, because the removed all-ones output component is erased exactly by the final LayerNorm and initialization is preserved modulo that symmetry.

INTENDED_EDIT: Keep the two successful leading `fc2` quotient columns, parameterize the final column in the same zero-mean basis, and reproduce the successful 1603-parameter design’s conceptual dense initialization and random-number stream.

EVIDENCE: Two leading quotients achieved 99.98% accuracy, while quotienting the adjacent third column reached only 96.12% even with initialization preservation; testing a different hidden channel isolates whether the remaining failure is channel-specific optimization sensitivity rather than loss of functional capacity.

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
        self.in_features = in_features
        self.out_features = out_features
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))
        self.rest_weight = nn.Parameter(torch.empty(out_features, in_features - 3))
        self.last_coordinates = nn.Parameter(torch.empty(out_features - 1))

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Match the original Linear(10, 8) constructor's random draw so the
        # already-successful leading coordinates retain their initialization.
        conceptual_rest = torch.empty(out_features, in_features - 2)
        nn.init.kaiming_uniform_(conceptual_rest, a=math.sqrt(5))
        nn.init.normal_(self.first_coordinates, mean=0.0, std=0.02)
        self._set_rest(conceptual_rest)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * (value @ self.reflector) / self.reflector_norm_sq
        return value - projection.unsqueeze(-1) * self.reflector

    def _set_rest(self, conceptual_rest: torch.Tensor) -> None:
        with torch.no_grad():
            self.rest_weight.copy_(conceptual_rest[:, :-1])
            transformed_last = self._householder(conceptual_rest[:, -1])
            self.last_coordinates.copy_(transformed_last[1:])

    def reset_rest_parameters(self) -> None:
        conceptual_rest = self.rest_weight.new_empty(
            self.out_features, self.in_features - 2
        )
        nn.init.normal_(conceptual_rest, mean=0.0, std=0.02)
        self._set_rest(conceptual_rest)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        first = self._householder(first)
        last = F.pad(self.last_coordinates, (1, 0))
        last = self._householder(last)
        weight = torch.cat(
            (first.transpose(0, 1), self.rest_weight, last.unsqueeze(1)), dim=1
        )
        return F.linear(x, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
            module.reset_rest_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE