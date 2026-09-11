MECHANISM: Distributed diagonal attention-output residual gauge

HYPOTHESIS: Fixing `proj.weight[3,3]` will produce a 1564-parameter model with at least 99% accuracy because subtracting this coefficient from column 3 adds only a feature-uniform residual shift removed by downstream LayerNorms, while distributing the new constraint away from projection row 0.

INTENDED_EDIT: Store 58 learned projection weights, reconstruct the existing five anchors plus `proj.weight[3,3]` as zero, and canonicalize column 3 using output row 3.

EVIDENCE: Moving the column-1 anchor from row 0, which failed at 15.51%, to its diagonal location `[1,1]` achieved 100% accuracy at 1565 parameters; applying the same distributed diagonal strategy to the next unanchored odd column is the closest supported reduction.

<<<<<<< SEARCH
class ResidualGaugeLinear(nn.Module):
    """Projection with five weights fixed along feature-uniform residual gauges."""
=======
class ResidualGaugeLinear(nn.Module):
    """Projection with six weights fixed along feature-uniform residual gauges."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(
            torch.empty(out_features * in_features - 5)
        )
=======
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
                self.weight_rest[5:22],
                zero,
                self.weight_rest[22:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0

                offset = weight[0, 4].clone()
=======
                offset = weight[0, 2].clone()
                weight[:, 2].sub_(offset)
                weight[0, 2] = 0.0

                # Continue the successful distributed diagonal strategy for
                # the next unanchored input column.
                offset = weight[3, 3].clone()
                weight[:, 3].sub_(offset)
                weight[3, 3] = 0.0

                offset = weight[0, 4].clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[7:9],
                            flat[10:],
=======
                            flat[7:9],
                            flat[10:27],
                            flat[28:],
>>>>>>> REPLACE