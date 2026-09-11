MECHANISM: Order-preserving third pre-attention LayerNorm scale tie

HYPOTHESIS: Tying the sixth `ln1` scale to the third learned scale while preserving the two successful existing tie mappings will reduce the model to 1,578 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Store five `AnchoredLayerNorm` scales; reconstruct scale 5 from scale 2 while continuing to reconstruct scales 6 and 7 from scales 0 and 1.

EVIDENCE: The two-tie mapping achieved 99.92%, whereas the failed final-three reconstruction remapped both established ties and achieved 46.2%; this isolates the additional tie without disturbing the successful mappings or all-ones initial function.

<<<<<<< SEARCH
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases and two dynamically tied scales."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:2]))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
=======
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
>>>>>>> REPLACE