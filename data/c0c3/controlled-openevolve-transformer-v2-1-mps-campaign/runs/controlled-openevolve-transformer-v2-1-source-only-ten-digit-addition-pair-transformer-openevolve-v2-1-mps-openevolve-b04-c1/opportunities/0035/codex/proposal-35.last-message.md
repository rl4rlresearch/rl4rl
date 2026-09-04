MECHANISM: Single-channel MLP output-shift quotient

HYPOTHESIS: Constraining one `fc2` output column to the seven-dimensional zero-mean subspace will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because its discarded all-ones component contributes only an activation-dependent residual shift erased by the final LayerNorm.

INTENDED_EDIT: Replace `fc2` with an otherwise dense linear layer whose first output column uses an isotropic Householder parameterization with seven learned coordinates.

EVIDENCE: The analogous isotropic quotient of the MLP output bias achieved 99.92% accuracy at 1605 parameters, whereas positional quotients affected position-specific optimization; this tests the same successful final-residual MLP symmetry on one exchangeable hidden channel.

<<<<<<< SEARCH
class ShiftGaugeEmbedding(nn.Module):
=======
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


class ShiftGaugeEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
=======
        self.fc2 = OneColumnGaugeLinear(d_ff, d_model)
>>>>>>> REPLACE