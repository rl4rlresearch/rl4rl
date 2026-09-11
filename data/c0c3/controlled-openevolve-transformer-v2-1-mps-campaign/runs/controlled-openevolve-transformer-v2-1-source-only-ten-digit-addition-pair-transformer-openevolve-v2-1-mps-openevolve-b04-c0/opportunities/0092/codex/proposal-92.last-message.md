MECHANISM: Third nonzero query–key scale-gauge anchor

HYPOTHESIS: Fixing `qkv.weight[2,0]` at 0.02 with reciprocal scaling of head-0 key row 2 will reduce the model to 1581 parameters while retaining at least 99% accuracy, because it preserves initial attention scores and extends the scale-gauge mechanism that succeeded for query rows 0 and 1.

INTENDED_EDIT: Store 187 learned QKV weights, reconstruct weights 0 and 2 as zero and weights 4, 8, and 16 as 0.02, then initialize query row 2 and key row 10 with reciprocal scaling.

EVIDENCE: Nonzero scale anchoring at weight 4 achieved 99.79% with 1583 parameters, and adding the same function-preserving mechanism at weight 8 achieved 99.93% with 1582 parameters; applying it to the next head-0 query/key coordinate is the most direct test of another independent scale redundancy.

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 4))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[5:],
=======
                fixed,
                self.weight_rest[5:12],
                fixed,
                self.weight_rest[12:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                scale = 0.02 / weight[1, 0]
                weight[1].mul_(scale)
                weight[9].div_(scale)

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat((flat[1:2], flat[3:4], flat[5:8], flat[9:]))
                )
=======
                scale = 0.02 / weight[1, 0]
                weight[1].mul_(scale)
                weight[9].div_(scale)

                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                flat = weight.flatten()
                module.weight_rest.copy_(
                    torch.cat(
                        (
                            flat[1:2],
                            flat[3:4],
                            flat[5:8],
                            flat[9:16],
                            flat[17:],
                        )
                    )
                )
>>>>>>> REPLACE