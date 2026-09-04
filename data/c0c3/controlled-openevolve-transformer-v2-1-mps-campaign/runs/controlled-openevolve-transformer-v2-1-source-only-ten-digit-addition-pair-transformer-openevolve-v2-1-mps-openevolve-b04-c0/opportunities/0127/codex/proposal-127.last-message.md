MECHANISM: Head-0 query–key shear gauge with preserved query scale

HYPOTHESIS: Fixing `qkv.weight[3,4]` at zero will produce a 1560-parameter model with at least 99% accuracy because a shear of query row 3 against row 0 preserves its existing zero at `[3,0]`, leaves row 3’s scale unconstrained, and can be inverted in key row 8.

INTENDED_EDIT: Store 175 learned QKV weights, reconstruct `qkv.weight[3,4]` as zero, canonicalize it through a reciprocal query–key shear, and restore key row 8’s existing LayerNorm-null anchor.

EVIDENCE: The 1561-parameter design achieved 99.93% after a third query–key shear, while directly fixing query row 3’s remaining scale achieved only 76.75%; this adds another shear without constraining that scale.

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
                self.weight_rest[54:69],
                zero,
                self.weight_rest[69:84],
                zero,
                self.weight_rest[84:99],
                zero,
                self.weight_rest[99:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                shear = weight[3, 0] / weight[1, 0]
                weight[3].sub_(shear * weight[1])
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                # Eliminate a component-two coefficient from query row 1
=======
                shear = weight[3, 0] / weight[1, 0]
                weight[3].sub_(shear * weight[1])
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                # Row 0's zero component-zero coefficient makes it a pivot
                # that eliminates row 3's component-four coefficient without
                # constraining row 3's remaining reciprocal scale.
                shear = weight[3, 4] / weight[0, 4]
                weight[3].sub_(shear * weight[0])
                weight[8].add_(shear * weight[11])
                weight[3, 4] = 0.0

                # Restore the key-row anchor perturbed by the inverse shear.
                key_offset = weight[8, 3].clone()
                weight[8].sub_(key_offset)
                weight[8, 3] = 0.0

                # Eliminate a component-two coefficient from query row 1
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[21:24],
                            flat[25:32],
                            flat[33:40],
=======
                            flat[21:24],
                            flat[25:28],
                            flat[29:32],
                            flat[33:40],
>>>>>>> REPLACE