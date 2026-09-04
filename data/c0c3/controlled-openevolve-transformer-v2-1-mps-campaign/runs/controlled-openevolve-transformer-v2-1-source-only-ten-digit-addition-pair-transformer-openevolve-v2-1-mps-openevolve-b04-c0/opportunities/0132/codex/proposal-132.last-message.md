MECHANISM: Same-head robust-row residual gauge with reduced validation overhead

HYPOTHESIS: Fixing `proj.weight[1,3]` will produce a 1560-parameter model with at least 99% accuracy, while evaluating every 600 steps will avoid the timeout that prevented the prior identical gauge from yielding accuracy evidence.

INTENDED_EDIT: Store 57 learned projection weights, reconstruct `proj.weight[1,3]` as zero, canonicalize column 3 through output row 1, and halve intermediate validation frequency without changing the 5000 training updates or final evaluation.

EVIDENCE: The row-1 column-1 anchor achieved 100% accuracy at 1565 parameters, making row 1 the strongest demonstrated location for an odd-column projection gauge; the direct row-1 column-3 trial timed out rather than failing accuracy, and three consecutive 1560-parameter projection trials timing out motivates reducing evaluation overhead.

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
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:40],
                zero,
                self.weight_rest[40:],
=======
                self.weight_rest[3:5],
                zero,
                self.weight_rest[5:6],
                zero,
                self.weight_rest[6:39],
                zero,
                self.weight_rest[39:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Subtracting each input-column coefficient from every output
                # row adds only feature-uniform residual shifts. Subsequent
                # LayerNorms remove all five shifts exactly.
=======
                # Subtracting each input-column coefficient from every output
                # row adds only feature-uniform residual shifts. Subsequent
                # LayerNorms remove all seven shifts exactly.
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

                # Reuse output row 1 for the remaining odd component of the
                # first head while preserving the residual-shift gauge.
                offset = weight[1, 3].clone()
                weight[:, 3].sub_(offset)
                weight[1, 3] = 0.0

                offset = weight[0, 2].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[5:6],
                            flat[7:9],
                            flat[10:45],
                            flat[46:],
=======
                            flat[5:6],
                            flat[7:9],
                            flat[10:11],
                            flat[12:45],
                            flat[46:],
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=600)
>>>>>>> REPLACE