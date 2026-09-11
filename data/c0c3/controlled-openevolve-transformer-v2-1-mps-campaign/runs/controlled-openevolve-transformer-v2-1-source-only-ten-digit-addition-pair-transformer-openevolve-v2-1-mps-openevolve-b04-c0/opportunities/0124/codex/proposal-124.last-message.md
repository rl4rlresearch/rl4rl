MECHANISM: Cross-head component-two LayerNorm-null key gauge

HYPOTHESIS: Fixing `qkv.weight[10,3]` at zero will produce a 1562-parameter model with at least 99% accuracy because it is the head-0 counterpart of the successful `[14,3]` component-two anchor.

INTENDED_EDIT: Store one fewer QKV weight, reconstruct `qkv.weight[10,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row.

EVIDENCE: `qkv.weight[14,3]` retained 99.19% accuracy at 1563 parameters, while component-zero key anchors succeeded in both heads and component-two projection anchors also succeeded in both heads, making the matching head-0 component-two key anchor the closest supported reduction.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with seven zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 14))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with eight zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 15))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight_rest[46:56],
                zero,
                self.weight_rest[56:87],
                zero,
                self.weight_rest[87:102],
                zero,
                self.weight_rest[102:],
=======
                self.weight_rest[46:56],
                zero,
                self.weight_rest[56:71],
                zero,
                self.weight_rest[71:86],
                zero,
                self.weight_rest[86:101],
                zero,
                self.weight_rest[101:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                key_offset = weight[8, 3].clone()
                weight[8].sub_(key_offset)
                weight[8, 3] = 0.0

                scale = 0.02 / weight[4, 0]
=======
                key_offset = weight[8, 3].clone()
                weight[8].sub_(key_offset)
                weight[8, 3] = 0.0

                # Mirror the successful component-two key anchor from head 1
                # in the corresponding key row of head 0.
                key_offset = weight[10, 3].clone()
                weight[10].sub_(key_offset)
                weight[10, 3] = 0.0

                scale = 0.02 / weight[4, 0]
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[49:56],
                            flat[57:67],
                            flat[68:99],
                            flat[100:115],
                            flat[116:],
=======
                            flat[49:56],
                            flat[57:67],
                            flat[68:83],
                            flat[84:99],
                            flat[100:115],
                            flat[116:],
>>>>>>> REPLACE