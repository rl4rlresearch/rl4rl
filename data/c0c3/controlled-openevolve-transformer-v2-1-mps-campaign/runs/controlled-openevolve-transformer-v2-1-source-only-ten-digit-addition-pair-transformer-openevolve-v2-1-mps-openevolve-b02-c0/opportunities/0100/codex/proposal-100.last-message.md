MECHANISM: Order-preserving fourth pre-attention LayerNorm scale tie

HYPOTHESIS: Tying the fifth `ln1` scale to the fourth learned scale while preserving all three verified tie mappings will reduce the model to 1,577 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Store four `AnchoredLayerNorm` scales; reconstruct scale 4 from scale 3 while continuing to reconstruct scales 5, 6, and 7 from scales 2, 0, and 1.

EVIDENCE: The order-preserving third tie achieved 99.97% at 1,578 parameters, whereas remapping the established ties achieved only 46.2%; extending the successful mapping without disturbing it is the closest controlled reduction.

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
            (self.weight, self.weight[3:4], self.weight[2:3], self.weight[:2])
        )
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE