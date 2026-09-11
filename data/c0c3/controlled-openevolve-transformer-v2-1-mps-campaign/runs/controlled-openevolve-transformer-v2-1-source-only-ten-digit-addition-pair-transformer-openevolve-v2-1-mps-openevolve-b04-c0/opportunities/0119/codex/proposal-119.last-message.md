MECHANISM: Cross-head component-two attention-output residual gauge

HYPOTHESIS: Fixing `proj.weight[0,6]` alongside `[0,0]`, `[0,2]`, and `[0,4]` will produce a 1566-parameter model with at least 99% accuracy because column 6 is the second-head counterpart of the successful column-2 anchor, and its removal adds only a feature-uniform residual shift eliminated by downstream LayerNorms.

INTENDED_EDIT: Store 60 learned projection weights, reconstruct `proj.weight[0,0]`, `[0,2]`, `[0,4]`, and `[0,6]` as zeros, and canonicalize all four columns at initialization.

EVIDENCE: The component-two projection anchor at `[0,2]` achieved 99.99% while the cross-head component-zero anchor at `[0,4]` achieved 99.63%; `[0,6]` combines the successful component-two choice with the successful cross-head symmetry, unlike the failed component-one `[0,1]` anchor.

<<<<<<< SEARCH
class ResidualGaugeLinear(nn.Module):
    """Projection with three weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 3)
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
                self.weight_rest[2:],
            )
        ).view(self.out_features, self.in_features)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Subtracting each input-column coefficient from every output
                # row adds only feature-uniform residual shifts. Subsequent
                # LayerNorms remove all three shifts exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0

                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:]))
                )
=======
                # Subtracting each input-column coefficient from every output
                # row adds only feature-uniform residual shifts. Subsequent
                # LayerNorms remove all four shifts exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0

                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                offset = weight[0, 6].clone()
                weight[:, 6].sub_(offset)
                weight[0, 6] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:6], flat[7:]))
                )
>>>>>>> REPLACE