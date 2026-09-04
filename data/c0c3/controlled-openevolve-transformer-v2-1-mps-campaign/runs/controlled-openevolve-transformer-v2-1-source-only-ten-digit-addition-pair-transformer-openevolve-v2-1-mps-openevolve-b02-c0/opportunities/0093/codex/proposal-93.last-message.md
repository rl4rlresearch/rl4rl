MECHANISM: Pre-MLP LayerNorm scale absorption

HYPOTHESIS: Fixing a fifth `ln2` scale at one will reduce the model to 1,579 parameters while retaining at least 99% accuracy, because the following learned `fc1` weights can absorb that coordinate’s scale and the initialization remains functionally and RNG-identical.

INTENDED_EDIT: Store three rather than four learned `BiasAnchoredLayerNorm` scales and reconstruct its final five scales as ones.

EVIDENCE: The verified 1,580-parameter design achieves 99.89% accuracy with four `ln2` scales already fixed at one, while the analogous normalized-input gauge removed one `fc1` weight per row and previously retained 99.93%; this makes one more pre-MLP scale constraint an independent, initialization-preserving reduction.

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