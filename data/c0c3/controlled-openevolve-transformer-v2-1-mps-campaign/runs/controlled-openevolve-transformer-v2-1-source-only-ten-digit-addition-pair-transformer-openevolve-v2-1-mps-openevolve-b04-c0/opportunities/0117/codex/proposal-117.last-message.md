MECHANISM: Cross-head attention-output residual-shift gauge

HYPOTHESIS: Fixing `proj.weight[0,4]` alongside the verified `proj.weight[0,0]` anchor will produce a 1568-parameter model with at least 99% accuracy, because column 4 is the component-zero counterpart in the second attention head and its removal adds only a feature-uniform residual shift eliminated by downstream LayerNorms.

INTENDED_EDIT: Store 62 learned attention-output projection weights, reconstruct `proj.weight[0,0]` and `proj.weight[0,4]` as zeros, and canonicalize both columns at initialization.

EVIDENCE: The component-zero projection anchor at `proj.weight[0,0]` achieved 99.93% at 1569 parameters, while the adjacent component-one anchor at `[0,1]` collapsed to 15.51%; prior QKV evidence also showed component-zero anchors succeeding across both heads while component one failed, motivating the cross-head component-zero counterpart.

<<<<<<< SEARCH
class ResidualGaugeLinear(nn.Module):
    """Projection with one weight fixed along a feature-uniform residual gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
        self.bias = None

    def weight_tensor(self) -> torch.Tensor:
        zero = self.weight_rest.new_zeros(1)
        return torch.cat((zero, self.weight_rest)).view(
            self.out_features, self.in_features
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
            with torch.no_grad():
                # Subtracting one input-column coefficient from every output
                # row adds only a feature-uniform residual shift. Subsequent
                # LayerNorms remove that shift exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0
                module.weight_rest.copy_(weight.flatten()[1:])
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
            with torch.no_grad():
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
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE