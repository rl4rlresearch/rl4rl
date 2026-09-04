MECHANISM: Third attention-output residual-shift gauge

HYPOTHESIS: Fixing `proj.weight[0,2]` alongside the successful component-zero anchors will produce a 1567-parameter model with at least 99% accuracy, because the removed projection coefficient contributes only a feature-uniform residual shift, and component 2 previously tolerated a QKV gauge reduction while component 1 did not.

INTENDED_EDIT: Store 61 learned attention-output projection weights, reconstruct `proj.weight[0,0]`, `[0,2]`, and `[0,4]` as zeros, and canonicalize all three columns at initialization.

EVIDENCE: Projection anchors `[0,0]` and `[0,4]` achieved 99.63% at 1568 parameters, whereas `[0,1]` collapsed to 15.51%; separately, the component-2 QKV shear achieved 99.59%, making projection component 2 the most supported untested column.

<<<<<<< SEARCH
class ResidualGaugeLinear(nn.Module):
    """Projection with two weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 2)
        )
        self.bias = None

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat(
            (zero, self.weight_rest[:3], zero, self.weight_rest[3:])
        ).view(self.out_features, self.in_features)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Subtracting either input-column coefficient from every
                # output row adds only a feature-uniform residual shift.
                # Subsequent LayerNorms remove both shifts exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:4], flat[5:]))
                )
=======
                # Subtracting each input-column coefficient from every
                # output row adds only a feature-uniform residual shift.
                # Subsequent LayerNorms remove all three shifts exactly.
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
>>>>>>> REPLACE