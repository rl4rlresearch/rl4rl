MECHANISM: Second head-1 query–key coordinate scale-gauge anchor

HYPOTHESIS: Fixing `qkv.weight[5,0]` at 0.02 with reciprocal scaling of head-1 key row 5 will reduce the model to 1579 parameters while retaining at least 99% accuracy, because the first function-preserving nonzero scale anchor in head 1 achieved 99.95% and this applies the same gauge mechanism to its next coordinate.

INTENDED_EDIT: Store 185 learned QKV weights, reconstruct weights 0 and 2 as zero and weights 4, 8, 16, 32, and 40 as 0.02, then initialize query row 5 and key row 13 with reciprocal scaling.

EVIDENCE: The head-1 nonzero anchor at `qkv.weight[4,0]` met the requirement with 99.95% accuracy and 1580 parameters, whereas a zero anchor in that head reached only 73.90%; extending the successful function-preserving scale anchor is the most informative next reduction.

<<<<<<< SEARCH
    """QKV projection with two zero anchors and four query scale anchors."""
=======
    """QKV projection with two zero anchors and five query scale anchors."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 6))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[12:27],
                fixed,
                self.weight_rest[27:],
=======
                fixed,
                self.weight_rest[12:27],
                fixed,
                self.weight_rest[27:34],
                fixed,
                self.weight_rest[34:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[4, 0]
                weight[4].mul_(scale)
                weight[12].div_(scale)

                flat = weight.flatten()
=======
                scale = 0.02 / weight[4, 0]
                weight[4].mul_(scale)
                weight[12].div_(scale)

                scale = 0.02 / weight[5, 0]
                weight[5].mul_(scale)
                weight[13].div_(scale)

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[17:32],
                            flat[33:],
=======
                            flat[17:32],
                            flat[33:40],
                            flat[41:],
>>>>>>> REPLACE