MECHANISM: Partner-specific fourth pre-attention LayerNorm scale tie

HYPOTHESIS: Tying the fifth `ln1` scale to learned scale 1 will reduce the model to 1,577 parameters while retaining at least 99% accuracy, with all three verified tie mappings preserved.

INTENDED_EDIT: Store four `AnchoredLayerNorm` scales and reconstruct scale 4 from scale 1 while preserving mappings 5→2, 6→0, and 7→1.

EVIDENCE: The verified three-tie mapping achieved 99.97% at 1,578 parameters. Fourth ties to scales 3, 2, and 0 produced materially different results, making scale 1 the only remaining learned partner for a controlled partner-specific test.

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
            (
                self.weight,
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:2],
            )
        )
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE