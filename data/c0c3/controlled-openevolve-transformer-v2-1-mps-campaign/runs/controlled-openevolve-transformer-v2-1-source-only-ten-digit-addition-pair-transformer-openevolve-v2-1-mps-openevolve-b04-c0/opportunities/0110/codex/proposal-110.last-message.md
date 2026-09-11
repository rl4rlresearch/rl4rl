MECHANISM: Parallel stable-pivot head-0 query/key shear anchor

HYPOTHESIS: Fixing `qkv.weight[1,4]` at zero through a shear against query row 0 will produce a 1571-parameter model with at least 99% accuracy, because it exactly preserves attention scores and mirrors the successful row-2 shear while retaining row 1’s fixed scale coefficient.

INTENDED_EDIT: Store 180 learned QKV weights, reconstruct `qkv.weight[1,4]` as zero, and initialize it by shearing query row 1 against query row 0 while applying the inverse shear to key row 8.

EVIDENCE: The identical stable-pivot shear on query row 2 achieved 99.59% at 1572 parameters, while the analogous row-3 constraint failed; row 1 resembles the successful row 2 because its scale anchor remains unchanged by the row-0 shear.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with four zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 11))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with five zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 12))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                self.weight_rest[46:],
=======
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
                self.weight_rest[45:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[1, 0]
                weight[1].mul_(scale)
                weight[9].div_(scale)

                scale = 0.02 / weight[2, 0]
=======
                scale = 0.02 / weight[1, 0]
                weight[1].mul_(scale)
                weight[9].div_(scale)

                # Use query row 0's stable fixed coefficient to eliminate
                # qkv.weight[1, 4]. The reciprocal key shear preserves every
                # head-0 attention score and leaves row 1's scale anchor fixed.
                shear = weight[1, 4] / weight[0, 4]
                weight[1].sub_(shear * weight[0])
                weight[8].add_(shear * weight[9])
                weight[1, 4] = 0.0

                scale = 0.02 / weight[2, 0]
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:20],
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
=======
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:12],
                            flat[13:16],
                            flat[17:20],
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
>>>>>>> REPLACE