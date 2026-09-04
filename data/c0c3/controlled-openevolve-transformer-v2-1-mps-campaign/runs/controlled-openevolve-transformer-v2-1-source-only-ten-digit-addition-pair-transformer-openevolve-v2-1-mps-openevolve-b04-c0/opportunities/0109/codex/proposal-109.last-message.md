MECHANISM: Stable-pivot head-0 query/key shear gauge anchor

HYPOTHESIS: Fixing `qkv.weight[2,4]` at zero through a shear against query row 0 will produce a 1572-parameter model with at least 99% accuracy, because row 2 already tolerates its scale anchor and the transformation uses the stable fixed `qkv.weight[0,4] = 0.02` pivot while exactly preserving attention scores.

INTENDED_EDIT: Store 181 learned QKV weights, reconstruct `qkv.weight[2,4]` as zero, and initialize it by shearing query row 2 against query row 0 while applying the inverse shear to key row 8.

EVIDENCE: The first shear anchor achieved 99.69% at 1573 parameters, while repeated scale, norm, and second-shear constraints specifically targeting query row 3 failed. Applying the same stable row-0 shear to the previously successful scale-anchored row 2 tests whether the last failure was row-3-specific.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with three zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 10))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with four zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 11))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[12:19],
                zero,
                self.weight_rest[19:26],
                fixed,
                self.weight_rest[26:33],
                fixed,
                self.weight_rest[33:40],
                fixed,
                self.weight_rest[40:47],
                fixed,
                self.weight_rest[47:],
=======
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
                self.weight_rest[46:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                scale = 0.02 / weight[4, 0]
=======
                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                # Use query row 0's stable fixed coefficient to eliminate
                # qkv.weight[2, 4]. The reciprocal key shear preserves every
                # head-0 attention score and leaves row 2's scale anchor fixed.
                shear = weight[2, 4] / weight[0, 4]
                weight[2].sub_(shear * weight[0])
                weight[8].add_(shear * weight[10])
                weight[2, 4] = 0.0

                scale = 0.02 / weight[4, 0]
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[9:16],
                            flat[17:24],
                            flat[25:32],
=======
                            flat[9:16],
                            flat[17:20],
                            flat[21:24],
                            flat[25:32],
>>>>>>> REPLACE