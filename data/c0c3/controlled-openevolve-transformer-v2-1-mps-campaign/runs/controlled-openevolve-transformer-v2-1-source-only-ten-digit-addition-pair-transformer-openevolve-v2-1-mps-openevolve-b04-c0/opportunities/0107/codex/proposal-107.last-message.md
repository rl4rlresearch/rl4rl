MECHANISM: Attention-preserving query/key shear gauge anchor

HYPOTHESIS: Fixing `qkv.weight[3,0]` at zero through a shear against the already anchored query row 1 will produce a 1573-parameter model with at least 99% accuracy, because the reciprocal key-row shear exactly preserves attention scores without the ill-conditioned scaling used by prior failed row-3 anchors.

INTENDED_EDIT: Store 182 learned QKV weights, reconstruct `qkv.weight[3,0]` as zero, and shear query row 3 against query row 1 while applying the inverse shear to key row 9 during initialization.

EVIDENCE: The current 1574-parameter design achieved 99.95%, while fixed-scale and norm constraints involving head-0 query row 3 failed despite representing exact Q/K gauges. The successful fixed coefficient in query row 1 provides a stable nonzero pivot for a different, initialization-preserving shear parameterization.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 9))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with three zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[12:27],
                fixed,
                self.weight_rest[27:34],
                fixed,
                self.weight_rest[34:41],
                fixed,
                self.weight_rest[41:48],
                fixed,
                self.weight_rest[48:],
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
                        )
                    )
                )
=======
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                # Use query row 1 as a stable nonzero pivot to eliminate
                # qkv.weight[3, 0]. Apply the inverse shear to its matching
                # key row so every head-0 attention score is preserved.
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
                            flat[17:24],
                            flat[25:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
                        )
                    )
                )
>>>>>>> REPLACE