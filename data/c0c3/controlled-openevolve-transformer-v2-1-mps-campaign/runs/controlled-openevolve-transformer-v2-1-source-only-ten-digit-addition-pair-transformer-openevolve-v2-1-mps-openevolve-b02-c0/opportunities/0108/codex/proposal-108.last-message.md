MECHANISM: Fixed-gauge pre-attention LayerNorm scale absorption

HYPOTHESIS: Fixing scale 3 at its initial value will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because QKV weights can absorb that scale without coupling its gradients into a successfully shared learned scale.

INTENDED_EDIT: Store three learned `AnchoredLayerNorm` scales, fix scale 3 at one, and preserve the verified mappings 4→1, 5→2, 6→0, and 7→1.

EVIDENCE: The preserved four-tie mapping achieved 99.73% at 1,577 parameters, while additionally tying scale 3 to scale 1 fell to 42.85%; fixing scale 3 isolates whether harmful shared-gradient coupling, rather than loss of its independent parameter, caused that failure.

<<<<<<< SEARCH
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
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with two anchored biases, four tied scales, and one fixed scale."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 5))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight,
                self.weight.new_ones(1),
                self.weight[1:2],
                self.weight[2:3],
                self.weight[:2],
            )
        )
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)
>>>>>>> REPLACE