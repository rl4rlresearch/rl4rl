MECHANISM: Second attention-preserving head-0 query/key shear gauge anchor

HYPOTHESIS: Fixing `qkv.weight[3,4]` at zero through a shear against query row 0 will produce a 1572-parameter model with at least 99% accuracy, because the transformation exactly preserves attention scores and avoids scaling query row 3.

INTENDED_EDIT: Store 181 learned QKV weights, reconstruct `qkv.weight[3,4]` as zero, and initialize it by shearing query row 3 against anchored query row 0 while applying the inverse shear to key row 8.

EVIDENCE: The first row-3 shear anchor achieved 99.69% at 1573 parameters, whereas directly scaling row 3 through the same column-4 coefficient achieved only 39.61%; this tests another exact shear using the stable fixed `qkv.weight[0,4] = 0.02` pivot.

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
                zero,
                self.weight_rest[19:22],
                zero,
                self.weight_rest[22:25],
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
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                flat = weight.flatten()
=======
                weight[9].add_(shear * weight[11])
                weight[3, 0] = 0.0

                # Eliminate a second coefficient from query row 3 using query
                # row 0, whose column-0 zero preserves the preceding anchor.
                # Apply the inverse shear to key row 8 so attention is unchanged.
                shear = weight[3, 4] / weight[0, 4]
                weight[3].sub_(shear * weight[0])
                weight[8].add_(shear * weight[11])
                weight[3, 4] = 0.0

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[17:24],
                            flat[25:32],
                            flat[33:40],
=======
                            flat[17:24],
                            flat[25:28],
                            flat[29:32],
                            flat[33:40],
>>>>>>> REPLACE