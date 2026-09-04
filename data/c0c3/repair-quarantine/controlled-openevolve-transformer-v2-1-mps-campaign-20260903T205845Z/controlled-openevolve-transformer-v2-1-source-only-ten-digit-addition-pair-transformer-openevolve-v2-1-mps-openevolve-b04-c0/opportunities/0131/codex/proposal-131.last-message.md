MECHANISM: Stable-pivot query/key reciprocal shear

HYPOTHESIS: Fixing `qkv.weight[1,4]` through the same exact row-0-pivot shear already used for `[2,4]` will reduce the model to 1565 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store 178 learned QKV weights, reconstruct `qkv.weight[1,4]` as zero, and canonicalize initialization with the compensating key-row shear.

EVIDENCE: The verified 1566-parameter model reaches 99.87% accuracy while using the identical stable `qkv.weight[0,4]=0.02` pivot to eliminate `[2,4]`; this tests another exact within-head attention gauge instead of the output-projection and shared-bias anchors that failed at 1565 parameters.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with six zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 13))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with seven zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 14))
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                fixed,
                self.weight_rest[2:5],
                fixed,
                self.weight_rest[5:12],
                fixed,
                self.weight_rest[12:15],
                zero,
                self.weight_rest[15:18],
                zero,
                self.weight_rest[18:25],
                fixed,
                self.weight_rest[25:32],
                fixed,
                self.weight_rest[32:39],
                fixed,
                self.weight_rest[39:46],
                fixed,
                self.weight_rest[46:56],
                zero,
                self.weight_rest[56:87],
                zero,
                self.weight_rest[87:],
            )
        ).view(self.out_features, self.in_features)
=======
        return torch.cat(
            (
                zero,
                self.weight_rest[:1],
                zero,
                self.weight_rest[1:2],
                fixed,
                self.weight_rest[2:5],
                fixed,
                self.weight_rest[5:8],
                zero,
                self.weight_rest[8:11],
                fixed,
                self.weight_rest[11:14],
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
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Use query row 0's stable fixed coefficient to eliminate
                # qkv.weight[2, 4]. The reciprocal key shear preserves every
                # head-0 attention score and leaves row 2's scale anchor fixed.
                shear = weight[2, 4] / weight[0, 4]
                weight[2].sub_(shear * weight[0])
                weight[8].add_(shear * weight[10])
                weight[2, 4] = 0.0

                # Canonicalize the corresponding component-zero key row in
=======
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

                # Canonicalize the corresponding component-zero key row in
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[5:8],
                            flat[9:16],
                            flat[17:20],
=======
                            flat[5:8],
                            flat[9:12],
                            flat[13:16],
                            flat[17:20],
>>>>>>> REPLACE