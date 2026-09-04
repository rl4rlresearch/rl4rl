MECHANISM: Head-aligned fourth pre-attention LayerNorm scale tie

HYPOTHESIS: Tying the fifth `ln1` scale to the first learned scale will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because coordinates 0 and 4 occupy corresponding dimensions in the two attention heads and all three verified tie mappings remain unchanged.

INTENDED_EDIT: Store four `AnchoredLayerNorm` scales and reconstruct scale 4 from scale 0 while preserving scale mappings 5→2, 6→0, and 7→1.

EVIDENCE: The verified three-tie mapping achieved 99.97% at 1,578 parameters; unsuccessful fourth ties to scales 3 and 2 reached 52.63% and 68.85%, respectively, motivating a controlled test of the architecturally head-aligned scale-0 partner.

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
            (self.weight, self.weight[:1], self.weight[2:3], self.weight[:2])
        )
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE