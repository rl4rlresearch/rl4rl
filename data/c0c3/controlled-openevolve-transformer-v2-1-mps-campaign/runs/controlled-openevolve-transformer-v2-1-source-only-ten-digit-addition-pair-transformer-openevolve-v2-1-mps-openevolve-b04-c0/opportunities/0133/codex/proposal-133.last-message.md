MECHANISM: LayerNorm-null query gauge with reciprocal shear restoration

HYPOTHESIS: Fixing `qkv.weight[3,2]` at zero will produce a 1560-parameter model with at least 99% accuracy because the learned row-3 query bias preserves the LayerNorm-null degree of freedom, while the successful `qkv.weight[1,2]` anchor supplies a stable fixed-scale pivot that restores `qkv.weight[3,0]` without constraining row 3’s scale.

INTENDED_EDIT: Store 175 learned QKV weights, reconstruct `qkv.weight[3,2]` as zero, and canonicalize initialization using a LayerNorm-null row shift followed by an inverse query/key shear through row 1.

EVIDENCE: The current 1561-parameter design achieved 99.93% with `qkv.weight[1,2]` fixed and row 1 retaining its `0.02` scale anchor; using that row’s zero target coordinate enables a stable additional gauge, while avoiding the direct row-3 scale constraint that collapsed to 76.75%.

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
=======
                self.weight_rest[14:17],
                zero,
                self.weight_rest[17:18],
                zero,
                self.weight_rest[18:23],
                fixed,
                self.weight_rest[23:30],
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
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:67],
                            flat[68:83],
                            flat[84:99],
                            flat[100:115],
                            flat[116:],
                        )
                    )
                )
=======
                # Eliminate a component-two coefficient from query row 1
                # without constraining query row 3's remaining scale. The
                # inverse key shear preserves every head-0 attention score.
                shear = weight[1, 2] / weight[3, 2]
                weight[1].sub_(shear * weight[3])
                weight[11].add_(shear * weight[9])
                weight[1, 2] = 0.0

                # Shift query row 3 along LayerNorm's null direction to remove
                # its component-two coefficient. Shearing through row 1 then
                # restores row 3's existing zero without fixing its scale.
                offset = weight[3, 2].clone()
                weight[3].sub_(offset)
                shear = -offset / weight[1, 0]
                weight[3].sub_(shear * weight[1])
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0
                weight[3, 2] = 0.0

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
                            flat[21:24],
                            flat[25:26],
                            flat[27:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:67],
                            flat[68:83],
                            flat[84:99],
                            flat[100:115],
                            flat[116:],
                        )
                    )
                )
>>>>>>> REPLACE