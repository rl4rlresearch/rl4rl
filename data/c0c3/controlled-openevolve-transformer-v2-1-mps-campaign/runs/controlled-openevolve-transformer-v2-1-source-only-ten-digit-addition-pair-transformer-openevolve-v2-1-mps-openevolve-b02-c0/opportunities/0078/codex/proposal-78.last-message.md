MECHANISM: Zero-mean LayerNorm input gauge fixing

HYPOTHESIS: Anchoring the final `fc1` weight in every output row to zero will reduce the model from 1,597 to 1,585 parameters while retaining at least 99% accuracy, because bias-free `ln2` produces normalized activations whose pre-scale coordinates sum to zero and whose final scale is fixed at one.

INTENDED_EDIT: Replace `fc1` with a compact linear layer storing seven rather than eight weights per output, reconstruct its final column as zero, and initialize it to the same function distribution as a full linear layer on zero-mean inputs.

EVIDENCE: The 1,597-parameter design still achieved 99.27%, while a fifth fixed `ln2` scale collapsed to 68.93%; this motivates preserving the four-scale design and removing an independent exact downstream linear redundancy instead.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = OutputAnchoredLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
class NormalizedInputLinear(nn.Linear):
    """Linear map with the final input weight fixed under a zero-mean gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            full_weight[:, :-1] - full_weight[:, -1:]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.pad(self.weight, (0, 1)), self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = NormalizedInputLinear(d_model, d_ff)
        self.fc2 = OutputAnchoredLinear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
=======
        if isinstance(module, NormalizedInputLinear):
            full_weight = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(
                    full_weight[:, :-1] - full_weight[:, -1:]
                )
                if module.bias is not None:
                    module.bias.zero_()
        elif isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
>>>>>>> REPLACE