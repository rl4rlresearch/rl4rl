MECHANISM: Second LayerNorm-null key-row coefficient anchor

HYPOTHESIS: Fixing `qkv.weight[13,3]` at zero will produce a 1570-parameter model with at least 99% accuracy, because the same LayerNorm-null, softmax-invariant key-row gauge used successfully for row 12 applies independently to row 13.

INTENDED_EDIT: Store 179 learned QKV weights, reconstruct `qkv.weight[13,3]` as zero, and canonicalize its initialized key row by subtracting that coefficient from the full row.

EVIDENCE: Fixing `qkv.weight[12,3]` with the identical key-only canonicalization achieved 99.97% accuracy at 1571 parameters, making an adjacent independent key row the closest supported next reduction.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with five zero anchors and seven query scale anchors."""
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with six zero anchors and seven query scale anchors."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 12))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 13))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight_rest[46:88],
                zero,
                self.weight_rest[88:],
=======
                self.weight_rest[46:88],
                zero,
                self.weight_rest[88:95],
                zero,
                self.weight_rest[95:],
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

                key_offset = weight[13, 3].clone()
                weight[13].sub_(key_offset)
                weight[13, 3] = 0.0

                # Use query row 1 as a stable nonzero pivot to eliminate
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[57:99],
                            flat[100:],
=======
                            flat[57:99],
                            flat[100:107],
                            flat[108:],
>>>>>>> REPLACE