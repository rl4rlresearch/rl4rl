MECHANISM: Stable-pivot third query/key shear

HYPOTHESIS: Fixing `qkv.weight[3,4]` through the same row-0-pivot shear already verified for `[1,4]` and `[2,4]` will reduce the model to 1564 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store 177 learned QKV weights, reconstruct `qkv.weight[3,4]` as zero, and canonicalize initialization with the compensating key-row shear.

EVIDENCE: The 1565-parameter design achieved 99.88% accuracy after eliminating `[1,4]` and `[2,4]` with the fixed nonzero `qkv.weight[0,4]` pivot; `[3,4]` admits the identical exact within-head gauge and preserves the existing `[3,0]` anchor because `qkv.weight[0,0]` is zero.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with seven zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 14))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with eight zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 15))
>>>>>>> REPLACE

<<<<<<< SEARCH
                zero,
                self.weight_rest[14:17],
                zero,
                self.weight_rest[17:24],
                fixed,
                self.weight_rest[24:31],
                fixed,
                self.weight_rest[31:38],
                fixed,
                self.weight_rest[38:45],
                fixed,
                self.weight_rest[45:55],
                zero,
                self.weight_rest[55:86],
                zero,
                self.weight_rest[86:],
=======
                zero,
                self.weight_rest[14:17],
                zero,
                self.weight_rest[17:20],
                zero,
                self.weight_rest[20:23],
                fixed,
                self.weight_rest[23:30],
                fixed,
                self.weight_rest[30:37],
                fixed,
                self.weight_rest[37:44],
                fixed,
                self.weight_rest[44:54],
                zero,
                self.weight_rest[54:85],
                zero,
                self.weight_rest[85:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Use query row 0's stable fixed coefficient to eliminate two
                # coordinates. Reciprocal key shears preserve every head-0
                # attention score and leave both query scale anchors fixed.
                shear = weight[1, 4] / weight[0, 4]
                weight[1].sub_(shear * weight[0])
                weight[8].add_(shear * weight[9])
                weight[1, 4] = 0.0

                shear = weight[2, 4] / weight[0, 4]
                weight[2].sub_(shear * weight[0])
                weight[8].add_(shear * weight[10])
                weight[2, 4] = 0.0
=======
                # Use query row 0's stable fixed coefficient to eliminate three
                # coordinates. Reciprocal key shears preserve every head-0
                # attention score and leave the other query anchors fixed.
                shear = weight[1, 4] / weight[0, 4]
                weight[1].sub_(shear * weight[0])
                weight[8].add_(shear * weight[9])
                weight[1, 4] = 0.0

                shear = weight[2, 4] / weight[0, 4]
                weight[2].sub_(shear * weight[0])
                weight[8].add_(shear * weight[10])
                weight[2, 4] = 0.0

                shear = weight[3, 4] / weight[0, 4]
                weight[3].sub_(shear * weight[0])
                weight[8].add_(shear * weight[11])
                weight[3, 4] = 0.0
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[17:20],
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
=======
                            flat[17:20],
                            flat[21:24],
                            flat[25:28],
                            flat[29:32],
                            flat[33:40],
>>>>>>> REPLACE