MECHANISM: Distributed attention-output residual-shift gauge

HYPOTHESIS: Fixing `proj.weight[1,1]` will produce a 1565-parameter model with at least 99% accuracy because subtracting this coefficient from its full input column adds only a feature-uniform residual shift removed by downstream LayerNorms, while avoiding a fifth anchor in projection output row 0.

INTENDED_EDIT: Store 59 learned projection weights, reconstruct the existing four row-0 anchors plus `proj.weight[1,1]` as zero, and canonicalize column 1 using output row 1.

EVIDENCE: Four even-column row-0 projection anchors reached 99.96% at 1566 parameters, whereas `proj.weight[0,1]` collapsed to 15.51%; moving the column-1 gauge anchor to row 1 directly tests whether that failure came from concentrating another constraint in output row 0 rather than from removing the exact column gauge itself.

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
                self.weight_rest[1:2],
                zero,
                self.weight_rest[2:3],
                zero,
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Subtracting each input-column coefficient from every output
                # row adds only feature-uniform residual shifts. Subsequent
                # LayerNorms remove all four shifts exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0
=======
                # Subtracting each input-column coefficient from every output
                # row adds only feature-uniform residual shifts. Subsequent
                # LayerNorms remove all five shifts exactly.
                offset = weight[0, 0].clone()
                weight[:, 0].sub_(offset)
                weight[0, 0] = 0.0

                # Distribute the column-1 anchor to output row 1 instead of
                # placing another optimization constraint in output row 0.
                offset = weight[1, 1].clone()
                weight[:, 1].sub_(offset)
                weight[1, 1] = 0.0

                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:6], flat[7:]))
                )
=======
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:6],
                            flat[7:9],
                            flat[10:],
                        )
                    )
                )
>>>>>>> REPLACE