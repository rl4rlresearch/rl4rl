MECHANISM: Fourth head-1 query–key coordinate scale-gauge anchor

HYPOTHESIS: Fixing `qkv.weight[7,0]` at 0.02 with reciprocal scaling of head-1 key row 7 will produce a 1577-parameter model with at least 99% accuracy; this tests whether the failed fourth head-0 anchor was head-specific rather than a general four-anchor limit.

INTENDED_EDIT: Store 183 learned QKV weights, reconstruct weights 0 and 2 as zero and weights 4, 8, 16, 32, 40, 48, and 56 as 0.02, and initialize query row 7 and key row 15 with reciprocal scaling.

EVIDENCE: Three successive head-1 scale anchors achieved 99.95%, 99.89%, and 99.94%, while the fourth head-0 anchor achieved only 71.61%; applying the same fourth-coordinate test to head 1 directly distinguishes head-specific sensitivity from scale-gauge saturation.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and six query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 8))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 9))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[34:41],
                fixed,
                self.weight_rest[41:],
=======
                fixed,
                self.weight_rest[34:41],
                fixed,
                self.weight_rest[41:48],
                fixed,
                self.weight_rest[48:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[6, 0]
                weight[6].mul_(scale)
                weight[14].div_(scale)

                flat = weight.flatten()
=======
                scale = 0.02 / weight[6, 0]
                weight[6].mul_(scale)
                weight[14].div_(scale)

                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[41:48],
                            flat[49:],
=======
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
>>>>>>> REPLACE