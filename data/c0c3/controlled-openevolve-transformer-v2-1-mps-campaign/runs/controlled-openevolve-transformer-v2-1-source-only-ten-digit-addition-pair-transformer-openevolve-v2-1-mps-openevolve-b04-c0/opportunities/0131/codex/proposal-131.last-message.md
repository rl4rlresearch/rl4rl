MECHANISM: Cross-head robust-row residual-shift gauge

HYPOTHESIS: Fixing `proj.weight[1,7]` will produce a 1560-parameter model with at least 99% accuracy because column 7 has an exact feature-uniform residual gauge and output row 1 already supported the successful difficult odd-column anchor.

INTENDED_EDIT: Store 57 learned projection weights, reconstruct `proj.weight[1,7]` as zero alongside the six existing anchors, and canonicalize column 7 using output row 1.

EVIDENCE: The `[1,1]` row-1 anchor achieved 100% accuracy, while `[3,3]` fell to 68.25%; the `[5,7]` trial timed out without accuracy evidence, so testing column 7 at the empirically strongest odd-anchor row is the most informative next reduction.

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
        self.bias = None
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
        self.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:40],
                zero,
                self.weight_rest[40:],
=======
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:10],
                zero,
                self.weight_rest[10:39],
                zero,
                self.weight_rest[39:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                offset = weight[1, 1].clone()
                weight[:, 1].sub_(offset)
                weight[1, 1] = 0.0

                offset = weight[0, 2].clone()
=======
                offset = weight[1, 1].clone()
                weight[:, 1].sub_(offset)
                weight[1, 1] = 0.0

                # Reuse the robust row-1 gauge section for the remaining
                # component-three column of the second attention head.
                offset = weight[1, 7].clone()
                weight[:, 7].sub_(offset)
                weight[1, 7] = 0.0

                offset = weight[0, 2].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[7:9],
                            flat[10:45],
                            flat[46:],
=======
                            flat[7:9],
                            flat[10:15],
                            flat[16:45],
                            flat[46:],
>>>>>>> REPLACE