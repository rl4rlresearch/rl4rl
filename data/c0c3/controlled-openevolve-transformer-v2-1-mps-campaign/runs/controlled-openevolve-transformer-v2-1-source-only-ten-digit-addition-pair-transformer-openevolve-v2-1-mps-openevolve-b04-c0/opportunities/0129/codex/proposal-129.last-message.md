MECHANISM: Reused robust output-row residual-shift gauge

HYPOTHESIS: Fixing `proj.weight[1,3]` will produce a 1560-parameter model with at least 99% accuracy because column 3 has an exact feature-uniform residual gauge, while output row 1 previously supported the difficult odd-component `[1,1]` anchor at 100% accuracy.

INTENDED_EDIT: Store 57 learned projection weights, reconstruct `proj.weight[1,3]` as zero alongside the six existing anchors, and canonicalize column 3 using output row 1.

EVIDENCE: The diagonal column-3 anchor `[3,3]` failed at 68.25%, but relocating the column-1 anchor from failing `[0,1]` to `[1,1]` achieved 100%; reusing the empirically robust row 1 distinguishes a column-3 limitation from an anchor-row optimization problem.

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
                zero,
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:40],
                zero,
                self.weight_rest[40:],
=======
                zero,
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:6],
                zero,
                self.weight_rest[6:39],
                zero,
                self.weight_rest[39:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Distribute the column-1 anchor to output row 1 instead of
                # placing another optimization constraint in output row 0.
                offset = weight[1, 1].clone()
                weight[:, 1].sub_(offset)
                weight[1, 1] = 0.0

                offset = weight[0, 2].clone()
=======
                # Distribute the column-1 anchor to output row 1 instead of
                # placing another optimization constraint in output row 0.
                offset = weight[1, 1].clone()
                weight[:, 1].sub_(offset)
                weight[1, 1] = 0.0

                # Reuse the robust output row 1 for the remaining odd
                # component of head 0, avoiding the failed diagonal anchor.
                offset = weight[1, 3].clone()
                weight[:, 3].sub_(offset)
                weight[1, 3] = 0.0

                offset = weight[0, 2].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[7:9],
                            flat[10:45],
                            flat[46:],
=======
                            flat[7:9],
                            flat[10:11],
                            flat[12:45],
                            flat[46:],
>>>>>>> REPLACE