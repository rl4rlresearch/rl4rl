MECHANISM: Cross-head component-zero LayerNorm-null key gauge

HYPOTHESIS: Fixing `qkv.weight[8,3]` at zero will produce a 1570-parameter model with at least 99% accuracy, because it is the head-0 counterpart of the successful `qkv.weight[12,3]` anchor and removes the same softmax-invariant key-row direction.

INTENDED_EDIT: Store 179 learned QKV weights, reconstruct `qkv.weight[8,3]` as zero, and canonicalize its initialized key row after the existing score-preserving scale and shear transformations.

EVIDENCE: The component-zero key anchor `qkv.weight[12,3]` achieved 99.97% at 1571 parameters, while the component-one row-13 anchor failed; testing the corresponding component-zero row in the other head is the closest way to distinguish component-specific viability from a one-row anomaly.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with five zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 12))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with six zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 13))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[46:88],
                zero,
                self.weight_rest[88:],
=======
                fixed,
                self.weight_rest[46:56],
                zero,
                self.weight_rest[56:87],
                zero,
                self.weight_rest[87:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                weight[8].add_(shear * weight[10])
                weight[2, 4] = 0.0

                scale = 0.02 / weight[4, 0]
=======
                weight[8].add_(shear * weight[10])
                weight[2, 4] = 0.0

                # Canonicalize the corresponding component-zero key row in
                # head 0. At initialization LayerNorm has unit gain, so this
                # only adds a softmax-invariant constant to every key.
                key_offset = weight[8, 3].clone()
                weight[8].sub_(key_offset)
                weight[8, 3] = 0.0

                scale = 0.02 / weight[4, 0]
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[49:56],
                            flat[57:99],
                            flat[100:],
=======
                            flat[49:56],
                            flat[57:67],
                            flat[68:99],
                            flat[100:],
>>>>>>> REPLACE