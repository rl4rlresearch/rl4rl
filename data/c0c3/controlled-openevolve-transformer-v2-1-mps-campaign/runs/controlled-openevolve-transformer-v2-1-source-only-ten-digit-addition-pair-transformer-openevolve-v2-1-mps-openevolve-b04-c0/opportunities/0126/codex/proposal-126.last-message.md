MECHANISM: Additional head-0 query–key shear gauge

HYPOTHESIS: Fixing `qkv.weight[1,2]` at zero through a reciprocal query/key shear will produce a 1561-parameter model with at least 99% accuracy, because the 99.98%-accurate 1562-parameter design already contains two successful query–key shear anchors, and this component-two reduction leaves query row 3’s scale unconstrained unlike the failed eighth scale anchor.

INTENDED_EDIT: Store 176 learned QKV weights, reconstruct `qkv.weight[1,2]` as zero, and canonicalize initialization by shearing query row 1 against row 3 while applying the inverse shear to key row 11.

EVIDENCE: The current design achieved 99.98% with existing shear eliminations at `qkv.weight[2,4]` and `[3,0]`; component-two projection and key anchors also succeeded, while fixing the remaining query scale collapsed to 76.75%, motivating another shear rather than another scale constraint.

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
    """QKV projection with nine zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 16))
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
                fixed,
                self.weight_rest[5:6],
                zero,
                self.weight_rest[6:11],
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
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:20],
=======
                shear = weight[3, 0] / weight[1, 0]
                weight[3].sub_(shear * weight[1])
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                # Eliminate a component-two coefficient from query row 1
                # without constraining query row 3's remaining scale. The
                # inverse key shear preserves every head-0 attention score.
                shear = weight[1, 2] / weight[3, 2]
                weight[1].sub_(shear * weight[3])
                weight[11].add_(shear * weight[9])
                weight[1, 2] = 0.0

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:10],
                            flat[11:16],
                            flat[17:20],
>>>>>>> REPLACE