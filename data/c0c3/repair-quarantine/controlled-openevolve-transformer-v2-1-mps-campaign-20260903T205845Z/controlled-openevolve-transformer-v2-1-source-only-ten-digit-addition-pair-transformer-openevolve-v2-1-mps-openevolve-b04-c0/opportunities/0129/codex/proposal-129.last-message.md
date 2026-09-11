MECHANISM: Fifth attention-output residual-shift gauge

HYPOTHESIS: Fixing `proj.weight[0,3]` in addition to `[0,0]`, `[0,2]`, `[0,4]`, and `[0,6]` will yield a 1565-parameter model with at least 99% accuracy because subtracting that coefficient from every output row changes the residual stream only by a feature-uniform shift removed by downstream LayerNorms.

INTENDED_EDIT: Store 59 learned attention-output projection weights, reconstruct `proj.weight[0,3]` as zero, and canonicalize its column during initialization.

EVIDENCE: Four projection-column anchors currently achieve 99.87% accuracy at 1566 parameters; the exact same column-wise residual-shift construction motivates testing a remaining coordinate, while avoiding component 1, whose prior anchor collapsed to 15.51%.

<<<<<<< SEARCH
class ResidualGaugeLinear(nn.Module):
    """Projection with four weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 4)
        )
        self.bias = None

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                zero,
                self.weight_rest[2:3],
                zero,
                self.weight_rest[3:],
            )
        ).view(self.out_features, self.in_features)
=======
class ResidualGaugeLinear(nn.Module):
    """Projection with five weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 5)
        )
        self.bias = None

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                zero,
                zero,
                self.weight_rest[1:2],
                zero,
                self.weight_rest[2:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0

                offset = weight[0, 4].clone()
=======
                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0

                offset = weight[0, 3].clone()
                weight[:, 3].sub_(offset)
                weight[0, 3] = 0.0

                offset = weight[0, 4].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:6], flat[7:]))
                )
=======
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[5:6], flat[7:]))
                )
>>>>>>> REPLACE