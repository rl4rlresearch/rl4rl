MECHANISM: Final query-row reciprocal scale gauge

HYPOTHESIS: Fixing `qkv.weight[3,1]` at `0.02` with the reciprocal key-row scaling will reduce the model to 1563 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store 176 learned QKV weights, reconstruct `qkv.weight[3,1]` as fixed, and canonicalize initialization by scaling query row 3 and inversely scaling key row 11.

EVIDENCE: The verified 1564-parameter model reached 99.93% accuracy with seven identical query/key reciprocal scale anchors; row 3 is the only query row without one, and scaling it preserves its existing zero anchors at `[3,0]` and `[3,4]`.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with eight zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 15))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with eight zero anchors and eight query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 16))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                shear = weight[3, 0] / weight[1, 0]
                weight[3].sub_(shear * weight[1])
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                flat = weight.flatten()
=======
                shear = weight[3, 0] / weight[1, 0]
                weight[3].sub_(shear * weight[1])
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                # Fix the remaining query row's scale while preserving every
                # head-0 attention score and both existing row-3 zero anchors.
                scale = 0.02 / weight[3, 1]
                weight[3].mul_(scale)
                weight[11].div_(scale)

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[21:24],
                            flat[25:28],
                            flat[29:32],
=======
                            flat[21:24],
                            flat[26:28],
                            flat[29:32],
>>>>>>> REPLACE