MECHANISM: Third head-1 query–key coordinate scale-gauge anchor

HYPOTHESIS: Fixing `qkv.weight[6,0]` at 0.02 with reciprocal scaling of head-1 key row 6 will reduce the model to 1578 parameters while retaining at least 99% accuracy, because the first two function-preserving scale anchors in head 1 achieved 99.95% and 99.89%.

INTENDED_EDIT: Store 184 learned QKV weights, reconstruct weights 0 and 2 as zero and weights 4, 8, 16, 32, 40, and 48 as 0.02, then initialize query row 6 and key row 14 with reciprocal scaling.

EVIDENCE: Adding the second head-1 nonzero scale anchor reduced the model to 1579 parameters while achieving 99.89%; extending the same initialization-preserving gauge mechanism to the next coordinate is the most direct capacity test.

<<<<<<< SEARCH
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and five query scale anchors."""
=======
class AttentionWeightAnchoredLinear(nn.Module):
    """QKV projection with two zero anchors and six query scale anchors."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 7))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 8))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[27:34],
                fixed,
                self.weight_rest[34:],
=======
                fixed,
                self.weight_rest[27:34],
                fixed,
                self.weight_rest[34:41],
                fixed,
                self.weight_rest[41:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[5, 0]
                weight[5].mul_(scale)
                weight[13].div_(scale)

                flat = weight.flatten()
=======
                scale = 0.02 / weight[5, 0]
                weight[5].mul_(scale)
                weight[13].div_(scale)

                scale = 0.02 / weight[6, 0]
                weight[6].mul_(scale)
                weight[14].div_(scale)

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[33:40],
                            flat[41:],
=======
                            flat[33:40],
                            flat[41:48],
                            flat[49:],
>>>>>>> REPLACE