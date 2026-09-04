MECHANISM: Second-head component-two LayerNorm-null key gauge

HYPOTHESIS: Fixing `qkv.weight[14,3]` at zero will produce a 1563-parameter model with at least 99% accuracy because component-two projection anchors succeeded in both attention heads, while the same key-row canonicalization succeeded for component zero in both heads.

INTENDED_EDIT: Store 177 learned QKV weights, reconstruct `qkv.weight[14,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row.

EVIDENCE: The current design reached 99.96% at 1564 parameters; `qkv.weight[8,3]` and `[12,3]` established cross-head viability for component-zero key gauges, while projection columns 2 and 6 established cross-head viability for component two, making row 14 better supported than the failed component-one or component-three directions.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with six zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 13))
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with seven zero anchors and seven query scale anchors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Preserve the random-number consumption of nn.Linear construction.
        nn.Linear(in_features, out_features, bias=False)
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 14))
>>>>>>> REPLACE

<<<<<<< SEARCH
                zero,
                self.weight_rest[56:87],
                zero,
                self.weight_rest[87:],
=======
                zero,
                self.weight_rest[56:87],
                zero,
                self.weight_rest[87:102],
                zero,
                self.weight_rest[102:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                key_offset = weight[12, 3].clone()
                weight[12].sub_(key_offset)
                weight[12, 3] = 0.0

                # Use query row 1 as a stable nonzero pivot to eliminate
=======
                key_offset = weight[12, 3].clone()
                weight[12].sub_(key_offset)
                weight[12, 3] = 0.0

                # Apply the same key-shift gauge to component two of head 1,
                # whose matching projection component has remained robust.
                key_offset = weight[14, 3].clone()
                weight[14].sub_(key_offset)
                weight[14, 3] = 0.0

                # Use query row 1 as a stable nonzero pivot to eliminate
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[57:67],
                            flat[68:99],
                            flat[100:],
=======
                            flat[57:67],
                            flat[68:99],
                            flat[100:115],
                            flat[116:],
>>>>>>> REPLACE