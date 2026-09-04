MECHANISM: Incremental pre-attention LayerNorm bias absorption

HYPOTHESIS: Fixing a third `ln1` bias coordinate to zero will reduce the model from 1,581 to 1,580 parameters while retaining at least 99% accuracy, because query shifts remain absorbable by query bias, key shifts cancel in attention softmax, and value shifts remain absorbable by the attention projection bias up to its irrelevant common mode.

INTENDED_EDIT: Store five rather than six learned `AnchoredLayerNorm` bias coordinates and reconstruct its final three coordinates as zeros.

EVIDENCE: The immediately preceding identical extension from one to two fixed `ln1` bias coordinates achieved 99.92% accuracy at 1,581 parameters, making a third coordinate the closest controlled reduction after adjacent positional, `fc2`, and attention-weight constraints failed.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with three anchored biases and dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE