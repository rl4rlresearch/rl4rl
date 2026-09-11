MECHANISM: Independent-head nonzero query–key scale-gauge anchor

HYPOTHESIS: Fixing `qkv.weight[4,0]` at 0.02 with reciprocal scaling of head-1 key row 4 will reduce the model to 1580 parameters while retaining at least 99% accuracy, because it preserves initial attention scores and uses an independent head’s coordinate-scale gauge rather than the head-0 coordinate whose fourth anchor failed.

INTENDED_EDIT: Store 186 learned QKV weights, reconstruct weights 0 and 2 as zero and weights 4, 8, 16, and 32 as 0.02, then initialize query row 4 and key row 12 with reciprocal scaling.

EVIDENCE: Three successive nonzero head-0 scale anchors passed at 99.79%, 99.93%, and 99.86%, while a zero anchor in head 1 reached only 73.90%; testing a function-preserving nonzero anchor in that independent head directly distinguishes loss of the query coefficient from head-specific optimization sensitivity.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and two query scale anchors."""
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and four query scale anchors."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 5))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[12:],
=======
                fixed,
                self.weight_rest[12:27],
                fixed,
                self.weight_rest[27:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                flat = weight.flatten()
=======
                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                scale = 0.02 / weight[4, 0]
                weight[4].mul_(scale)
                weight[12].div_(scale)

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[9:16],
                            flat[17:],
=======
                            flat[9:16],
                            flat[17:32],
                            flat[33:],
>>>>>>> REPLACE