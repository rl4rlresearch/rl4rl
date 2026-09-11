MECHANISM: Completion of the query–key reciprocal scale gauge

HYPOTHESIS: Fixing `qkv.weight[3,1]` at `0.02` while inversely scaling key row 11 will produce a 1561-parameter model with at least 99% accuracy, because it uses the eighth exact query/key scale symmetry already exploited by seven successful anchors.

INTENDED_EDIT: Store 176 learned QKV weights, reconstruct `qkv.weight[3,1]` as fixed, and canonicalize its initialization after the existing row-3 shear through reciprocal query/key scaling.

EVIDENCE: The current 1562-parameter model achieved 99.98% accuracy while using seven query-scale anchors; the remaining query row already has a preserved zero anchor and an unconstrained reciprocal scale gauge.

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
                self.weight_rest[56:71],
                zero,
                self.weight_rest[71:86],
                zero,
                self.weight_rest[86:101],
                zero,
                self.weight_rest[101:],
=======
                self.weight_rest[12:15],
                zero,
                self.weight_rest[15:18],
                zero,
                fixed,
                self.weight_rest[18:24],
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

                # Fix the remaining query-row scale while inversely scaling
                # its matching key row, preserving every attention score.
                scale = 0.02 / weight[3, 1]
                weight[3].mul_(scale)
                weight[11].div_(scale)

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[17:20],
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
=======
                            flat[17:20],
                            flat[21:24],
                            flat[26:32],
                            flat[33:40],
>>>>>>> REPLACE