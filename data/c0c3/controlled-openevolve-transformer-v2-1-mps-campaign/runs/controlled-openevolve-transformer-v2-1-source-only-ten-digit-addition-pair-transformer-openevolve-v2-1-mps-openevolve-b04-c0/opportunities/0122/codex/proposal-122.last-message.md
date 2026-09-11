MECHANISM: Cross-head diagonal attention-output residual gauge

HYPOTHESIS: Fixing `proj.weight[5,5]` will produce a 1564-parameter model with at least 99% accuracy because it is the second-head counterpart of the successful distributed `[1,1]` anchor, while avoiding the component-three direction implicated by the failed `[3,3]` experiment.

INTENDED_EDIT: Store 58 learned projection weights, reconstruct the existing five anchors plus `proj.weight[5,5]` as zero, and canonicalize column 5 using output row 5.

EVIDENCE: Moving the column-1 anchor from `[0,1]`, which achieved only 15.51%, to `[1,1]` achieved 100% accuracy at 1565 parameters; testing the matching diagonal component-one anchor in the second head is better supported than repeating the component-three strategy that reached only 68.25%.

<<<<<<< SEARCH
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
=======
class ResidualGaugeLinear(nn.Module):
    """Projection with six weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 6)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:],
=======
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:40],
                zero,
                self.weight_rest[40:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                offset = weight[0, 6].clone()
                weight[:, 6].sub_(offset)
                weight[0, 6] = 0.0

                flat = weight.flatten()
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
=======
                offset = weight[0, 4].clone()
                weight[:, 4].sub_(offset)
                weight[0, 4] = 0.0

                # Mirror the successful distributed column-1 anchor in the
                # corresponding component of the second attention head.
                offset = weight[5, 5].clone()
                weight[:, 5].sub_(offset)
                weight[5, 5] = 0.0

                offset = weight[0, 6].clone()
                weight[:, 6].sub_(offset)
                weight[0, 6] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:6],
                            flat[7:9],
                            flat[10:45],
                            flat[46:],
                        )
                    )
                )
>>>>>>> REPLACE