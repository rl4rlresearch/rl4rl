MECHANISM: Partner-specific fourth pre-attention LayerNorm scale tie

HYPOTHESIS: Tying the fifth `ln1` scale to the third learned scale will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because that learned scale already supports the verified sixth-scale tie without disturbing the three successful mappings.

INTENDED_EDIT: Store four `AnchoredLayerNorm` scales and reconstruct scales 4 and 5 from scale 2 while preserving the verified mappings for scales 6 and 7.

EVIDENCE: The order-preserving third tie to scale 2 achieved 99.97% at 1,578 parameters, whereas tying scale 4 to scale 3 achieved only 52.63%; changing only the new tie’s partner directly tests whether that failure was partner-specific.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and three dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 3))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[2:3], self.weight[:2]))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and four dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[2:3], self.weight[2:3], self.weight[:2])
        )
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE