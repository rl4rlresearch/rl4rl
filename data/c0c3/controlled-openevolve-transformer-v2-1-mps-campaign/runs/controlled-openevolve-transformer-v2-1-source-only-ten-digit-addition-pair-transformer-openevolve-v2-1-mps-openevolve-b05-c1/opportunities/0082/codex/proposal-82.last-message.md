MECHANISM: Single-column orthonormal attention-output quotient

HYPOTHESIS: Removing only one row-common attention-output degree of freedom will reduce the model from 1,272 to 1,271 parameters while retaining at least 99% accuracy, because the discarded component produces only a LayerNorm-invisible all-ones residual shift, and limiting the quotient reparameterization to one column avoids the global optimization disruption seen when all eight columns were changed.

INTENDED_EDIT: Parameterize the first attention-output weight column in a seven-dimensional orthonormal mean-zero basis while leaving the other seven columns dense and retaining the full bias.

EVIDENCE: The current 1,272-parameter model reached 100% accuracy, while removing all eight attention-output common modes reached only 74.64% with an oblique chart and 41.25% with an orthonormal chart. A one-column quotient is the smallest test of whether those failures came from simultaneously reparameterizing the entire projection rather than from removing a single redundant direction.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with five fixed scales and seven fixed bias coordinates."""
=======
class OneColumnQuotientLinear(nn.Module):
    """Linear map with one output column represented modulo common shifts."""

    def __init__(self, features: int):
        super().__init__()
        self.first_column = nn.Parameter(torch.empty(features - 1))
        self.other_columns = nn.Parameter(torch.empty(features, features - 1))
        self.bias = nn.Parameter(torch.empty(features))

        basis = torch.zeros(features, features - 1)
        for col in range(features - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("output_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_column = self.output_basis @ self.first_column
        weight = torch.cat(
            (first_column.unsqueeze(1), self.other_columns),
            dim=1,
        )
        return F.linear(x, weight, self.bias)


class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with five fixed scales and seven fixed bias coordinates."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
=======
        self.proj = OneColumnQuotientLinear(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OneColumnQuotientLinear):
            nn.init.normal_(module.first_column, mean=0.0, std=0.02)
            nn.init.normal_(module.other_columns, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE