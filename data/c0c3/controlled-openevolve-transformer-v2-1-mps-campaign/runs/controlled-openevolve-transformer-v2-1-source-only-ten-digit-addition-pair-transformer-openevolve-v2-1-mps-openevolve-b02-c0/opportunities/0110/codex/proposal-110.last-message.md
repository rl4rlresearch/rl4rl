MECHANISM: Adjacent pre-MLP LayerNorm scale absorption

HYPOTHESIS: Fixing `ln2` scale coordinate 3 at one will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because `fc1` can absorb that scale and initialization remains functionally and RNG-identical.

INTENDED_EDIT: Store three learned `BiasAnchoredLayerNorm` scales and fix its final five scales at one.

EVIDENCE: The verified 1,577-parameter model achieves 99.73% accuracy while already fixing four adjacent `ln2` scales; extending this established gauge is more directly supported than further constraining `ln1`, attention output, or positional embeddings, whose latest one-parameter reductions failed.

<<<<<<< SEARCH
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final four scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with zero bias and its final five scales fixed at one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 5), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None, 1e-5)
>>>>>>> REPLACE