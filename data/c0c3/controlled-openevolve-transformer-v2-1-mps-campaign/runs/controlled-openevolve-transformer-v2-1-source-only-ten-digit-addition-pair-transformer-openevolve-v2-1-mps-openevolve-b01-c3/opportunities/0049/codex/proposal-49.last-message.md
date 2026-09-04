MECHANISM: Axis-aligned LayerNorm-nullspace anchoring in one value row

HYPOTHESIS: A 1,402-parameter model will retain at least 99% accuracy because anchoring the final value-projection row on the mean-zero LayerNorm input removes a functionally redundant coefficient, while an axis-aligned parameterization and distribution-matched initialization avoid the optimization disruption seen with orthonormal nullspace bases.

INTENDED_EDIT: Replace QKV with a 191-parameter projection whose final value row has seven learned coordinates and a zero eighth coordinate, initialized to reproduce the original row’s function on mean-zero inputs.

EVIDENCE: The current 1,403-parameter model achieved 99.98%. Prior LayerNorm-nullspace reductions preserved the theoretical function class but fell to 84.82% globally and 96.78% for one orthonormally parameterized MLP row, motivating a single, axis-aligned reduction in the less routing-sensitive value projection.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""
=======
class LastValueRowAnchoredLinear(nn.Module):
    """Bias-free linear map with the final row anchored on mean-zero inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features < 2 or out_features < 1:
            raise ValueError("invalid linear dimensions")
        self.weight = nn.Parameter(torch.empty(out_features - 1, in_features))
        self.last_weight = nn.Parameter(torch.empty(in_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_row = F.pad(self.last_weight, (0, 1))
        weight = torch.cat((self.weight, last_row.unsqueeze(0)), dim=0)
        return F.linear(x, weight)


class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = LastValueRowAnchoredLinear(d_model, 3 * d_model)
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
        if isinstance(module, LastValueRowAnchoredLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                full_row = module.last_weight.new_empty(module.last_weight.numel() + 1)
                nn.init.normal_(full_row, mean=0.0, std=0.02)
                module.last_weight.copy_(full_row[:-1] - full_row[-1])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE