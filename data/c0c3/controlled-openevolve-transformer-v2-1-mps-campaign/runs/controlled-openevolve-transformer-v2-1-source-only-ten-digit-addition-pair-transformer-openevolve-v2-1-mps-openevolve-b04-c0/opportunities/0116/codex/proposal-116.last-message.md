MECHANISM: Second attention-output residual-shift weight gauge

HYPOTHESIS: Fixing `proj.weight[0,1]` at zero alongside the verified `proj.weight[0,0]` anchor will produce a 1568-parameter model with at least 99% accuracy, because independently subtracting each coefficient from its full output column changes every token only by a feature-uniform shift removed by downstream LayerNorms.

INTENDED_EDIT: Store 62 learned attention-output projection weights, reconstruct `proj.weight[0,0:2]` as zeros, and canonicalize both corresponding columns at initialization.

EVIDENCE: The first attention-output residual-shift anchor achieved 99.93% accuracy with 1569 parameters; applying the identical exact gauge to the adjacent input column is the closest supported one-parameter reduction.

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
        zeros = self.weight_rest.new_zeros(2)
        return torch.cat((zeros, self.weight_rest)).view(
            self.out_features, self.in_features
        )
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
                # Subtracting each input-column coefficient from every output
                # row adds only feature-uniform residual shifts. Subsequent
                # LayerNorms remove those shifts exactly.
                offsets = weight[0, :2].clone()
                weight[:, :2].sub_(offsets)
                weight[0, :2] = 0.0
                module.weight_rest.copy_(weight.flatten()[2:])
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE