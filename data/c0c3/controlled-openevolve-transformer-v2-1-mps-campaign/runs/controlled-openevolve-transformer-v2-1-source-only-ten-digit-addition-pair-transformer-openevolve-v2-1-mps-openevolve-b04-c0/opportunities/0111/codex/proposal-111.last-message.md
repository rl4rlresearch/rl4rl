MECHANISM: LayerNorm-null key-row coefficient anchor

HYPOTHESIS: Fixing `qkv.weight[12,3]` at zero will produce a 1571-parameter model with at least 99% accuracy, because a LayerNorm output has a zero-sum standardized component, so the removed key-row coefficient is gauge-redundant up to a position-independent key shift that causal softmax attention cancels.

INTENDED_EDIT: Store 180 learned QKV weights, reconstruct `qkv.weight[12,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row, preserving the initialized attention function.

EVIDENCE: The current 1572-parameter model reached 99.59%, while the next query-shear reduction collapsed to 41.62%; this motivates using an untested additive key-only invariance that requires neither another query shear nor reciprocal rescaling.

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
                self.weight_rest[39:46],
                fixed,
                self.weight_rest[46:],
=======
                fixed,
                self.weight_rest[39:46],
                fixed,
                self.weight_rest[46:88],
                zero,
                self.weight_rest[88:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                # Use query row 1 as a stable nonzero pivot to eliminate
=======
                scale = 0.02 / weight[7, 0]
                weight[7].mul_(scale)
                weight[15].div_(scale)

                # Canonicalize one key row along LayerNorm's null direction.
                # The resulting position-independent key shift cancels in
                # every attention softmax.
                key_offset = weight[12, 3].clone()
                weight[12].sub_(key_offset)
                weight[12, 3] = 0.0

                # Use query row 1 as a stable nonzero pivot to eliminate
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[49:56],
                            flat[57:],
=======
                            flat[49:56],
                            flat[57:99],
                            flat[100:],
>>>>>>> REPLACE