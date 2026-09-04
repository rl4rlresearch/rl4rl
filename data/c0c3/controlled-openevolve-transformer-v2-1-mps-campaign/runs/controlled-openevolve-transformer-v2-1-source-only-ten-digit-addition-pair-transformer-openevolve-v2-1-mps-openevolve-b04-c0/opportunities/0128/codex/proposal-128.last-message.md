MECHANISM: Head-1 scale-preserving reciprocal query–key shear

HYPOTHESIS: Fixing `qkv.weight[4,2]` at zero will produce a 1560-parameter model with at least 99% accuracy because it mirrors the successful `[0,2]` anchor while preserving all head-1 query scales and attention scores.

INTENDED_EDIT: Store 175 learned QKV weights, reconstruct `qkv.weight[4,2]` as zero, and canonicalize it by shearing query row 4 against the scale-neutral difference of rows 5 and 7 with inverse updates to key rows 13 and 15.

EVIDENCE: The 1561-parameter design retained 99.93% with `qkv.weight[0,2]` and an additional component-two shear; cross-head component-two key and projection anchors also succeeded, while the failed `[3,4]` experiment motivates leaving query row 3 unconstrained.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with nine zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 16))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with ten zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 17))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[24:31],
                fixed,
                self.weight_rest[31:38],
                fixed,
                self.weight_rest[38:45],
                fixed,
                self.weight_rest[45:55],
                zero,
                self.weight_rest[55:70],
                zero,
                self.weight_rest[70:85],
                zero,
                self.weight_rest[85:100],
                zero,
                self.weight_rest[100:],
=======
                fixed,
                self.weight_rest[24:25],
                zero,
                self.weight_rest[25:30],
                fixed,
                self.weight_rest[30:37],
                fixed,
                self.weight_rest[37:44],
                fixed,
                self.weight_rest[44:54],
                zero,
                self.weight_rest[54:69],
                zero,
                self.weight_rest[69:84],
                zero,
                self.weight_rest[84:99],
                zero,
                self.weight_rest[99:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                key_offset = weight[14, 3].clone()
                weight[14].sub_(key_offset)
                weight[14, 3] = 0.0

                # Use query row 1 as a stable nonzero pivot to eliminate
=======
                key_offset = weight[14, 3].clone()
                weight[14].sub_(key_offset)
                weight[14, 3] = 0.0

                # Rows 5 and 7 have the same fixed scale coefficient, so their
                # difference is scale-neutral. Shear row 4 along that
                # difference and apply the inverse transformation to the two
                # unanchored matching key rows.
                shear = weight[4, 2] / (weight[5, 2] - weight[7, 2])
                weight[4].sub_(shear * (weight[5] - weight[7]))
                weight[13].add_(shear * weight[12])
                weight[15].sub_(shear * weight[12])
                weight[4, 2] = 0.0

                # Use query row 1 as a stable nonzero pivot to eliminate
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
                            flat[41:48],
=======
                            flat[21:24],
                            flat[25:32],
                            flat[33:34],
                            flat[35:40],
                            flat[41:48],
>>>>>>> REPLACE