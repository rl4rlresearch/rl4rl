MECHANISM: Second MLP hidden-channel shift quotient

HYPOTHESIS: Constraining a second `fc2` weight column to the seven-dimensional zero-mean subspace will reduce the model from 1604 to 1603 parameters while retaining at least 99% accuracy, because each discarded all-ones component independently produces only a token-wise residual shift erased by the final LayerNorm.

INTENDED_EDIT: Extend `OneColumnGaugeLinear` to parameterize its first two weight columns with independent Householder zero-mean coordinates, leaving the other ten columns dense.

EVIDENCE: The identical quotient applied to one `fc2` column achieved 99.69% accuracy at 1604 parameters; extending that successful, exact final-residual symmetry by one additional exchangeable hidden channel is the most direct test.

<<<<<<< SEARCH
class OneColumnGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.rest = nn.Linear(in_features - 1, out_features, bias=False)
        self.first_coordinates = nn.Parameter(torch.empty(out_features - 1))

        inv_sqrt = out_features ** -0.5
        reflector = torch.full((out_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
        nn.init.normal_(self.first_coordinates, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = F.pad(self.first_coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, first) / self.reflector_norm_sq
        first = first - self.reflector * projection
        weight = torch.cat((first.unsqueeze(1), self.rest.weight), dim=1)
        return F.linear(x, weight)
=======
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
>>>>>>> REPLACE