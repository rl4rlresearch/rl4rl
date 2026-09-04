MECHANISM: Second-head robust-row residual-shift gauge

HYPOTHESIS: Fixing `proj.weight[5,7]` will produce a 1560-parameter model with at least 99% accuracy because column 7 has the same exact feature-uniform residual gauge as the successful `[5,5]` anchor, while retaining the empirically robust output row 5 and avoiding the failed output row 3.

INTENDED_EDIT: Store 57 learned projection weights, reconstruct `proj.weight[5,7]` as zero alongside the six existing anchors, and canonicalize column 7 using output row 5.

EVIDENCE: `proj.weight[5,5]` retained 99.96% accuracy at 1564 parameters, and the current 1561-parameter design reached 99.93%; using row 5 for the remaining second-head column tests the closest robust-row analogue while the failed `[3,3]` result argues against output row 3.

<<<<<<< SEARCH
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
=======
class ResidualGaugeLinear(nn.Module):
    """Projection with seven weights fixed along feature-uniform residual gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=True)
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 7)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight_rest[5:40],
                zero,
                self.weight_rest[40:],
=======
                self.weight_rest[5:40],
                zero,
                self.weight_rest[40:41],
                zero,
                self.weight_rest[41:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Mirror the successful distributed column-1 anchor in the
                # corresponding component of the second attention head.
                offset = weight[5, 5].clone()
                weight[:, 5].sub_(offset)
                weight[5, 5] = 0.0

                offset = weight[0, 6].clone()
=======
                # Mirror the successful distributed column-1 anchor in the
                # corresponding component of the second attention head.
                offset = weight[5, 5].clone()
                weight[:, 5].sub_(offset)
                weight[5, 5] = 0.0

                # Reuse robust output row 5 for the remaining input component
                # of the second head.
                offset = weight[5, 7].clone()
                weight[:, 7].sub_(offset)
                weight[5, 7] = 0.0

                offset = weight[0, 6].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[7:9],
                            flat[10:45],
                            flat[46:],
=======
                            flat[7:9],
                            flat[10:45],
                            flat[46:47],
                            flat[48:],
>>>>>>> REPLACE