MECHANISM: Stable-pivot cross-query shear gauge

HYPOTHESIS: Fixing `qkv.weight[1,1]` at zero through query row 3’s verified `0.02` pivot will reduce the model to 1562 parameters while retaining at least 99% accuracy, because the compensating key-row shear preserves every attention score and does not disturb existing row-1 anchors.

INTENDED_EDIT: Store 175 learned QKV weights, reconstruct `qkv.weight[1,1]` as zero, and canonicalize initialization by shearing query row 1 against row 3 while applying the inverse shear to key row 11.

EVIDENCE: The current 1563-parameter model achieved 99.74% accuracy after fixing `qkv.weight[3,1]` at `0.02`; row 3 also has verified zeros at columns 0 and 4, so it can eliminate row 1 column 1 while preserving row 1’s fixed column-0 scale and column-4 zero.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with eight zero anchors and eight query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 16))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with nine zero anchors and eight query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 17))
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
                self.weight_rest[5:8],
                zero,
                self.weight_rest[8:11],
                fixed,
                self.weight_rest[11:14],
                zero,
                self.weight_rest[14:17],
                zero,
                fixed,
                self.weight_rest[17:19],
                zero,
                self.weight_rest[19:22],
                fixed,
                self.weight_rest[22:29],
                fixed,
                self.weight_rest[29:36],
                fixed,
                self.weight_rest[36:43],
                fixed,
                self.weight_rest[43:53],
                zero,
                self.weight_rest[53:84],
                zero,
                self.weight_rest[84:],
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
                zero,
                self.weight_rest[5:7],
                zero,
                self.weight_rest[7:10],
                fixed,
                self.weight_rest[10:13],
                zero,
                self.weight_rest[13:16],
                zero,
                fixed,
                self.weight_rest[16:18],
                zero,
                self.weight_rest[18:21],
                fixed,
                self.weight_rest[21:28],
                fixed,
                self.weight_rest[28:35],
                fixed,
                self.weight_rest[35:42],
                fixed,
                self.weight_rest[42:52],
                zero,
                self.weight_rest[52:83],
                zero,
                self.weight_rest[83:],
            )
        ).view(self.out_features, self.in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Fix the remaining query row's scale while preserving every
                # head-0 attention score and both existing row-3 zero anchors.
                scale = 0.02 / weight[3, 1]
                weight[3].mul_(scale)
                weight[11].div_(scale)

                flat = weight.flatten()
=======
                # Fix the remaining query row's scale while preserving every
                # head-0 attention score and both existing row-3 zero anchors.
                scale = 0.02 / weight[3, 1]
                weight[3].mul_(scale)
                weight[11].div_(scale)

                # Use row 3's stable component-one pivot to eliminate the
                # matching row-1 coordinate. Its existing component-zero and
                # component-four zeros preserve both anchors in query row 1.
                shear = weight[1, 1] / weight[3, 1]
                weight[1].sub_(shear * weight[3])
                weight[11].add_(shear * weight[9])
                weight[1, 1] = 0.0

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[5:8],
                            flat[9:12],
                            flat[13:16],
=======
                            flat[5:8],
                            flat[10:12],
                            flat[13:16],
>>>>>>> REPLACE