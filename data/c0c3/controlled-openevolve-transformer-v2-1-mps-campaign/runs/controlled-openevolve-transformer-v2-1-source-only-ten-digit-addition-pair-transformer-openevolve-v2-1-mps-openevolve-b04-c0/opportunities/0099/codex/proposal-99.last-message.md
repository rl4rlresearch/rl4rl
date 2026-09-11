MECHANISM: Alternative-coefficient query–key scale-gauge anchor

HYPOTHESIS: Fixing `qkv.weight[3,4]` at 0.02 with reciprocal scaling of head-0 key row 3 will produce a 1576-parameter model with at least 99% accuracy, because it uses the exact fourth head-0 coordinate-scale gauge without repeating the unsuccessful column-0 anchor.

INTENDED_EDIT: Store 182 learned QKV weights, reconstruct `qkv.weight[3,4]` as 0.02, and initialize query row 3 and key row 11 with reciprocal scaling that preserves initial attention scores.

EVIDENCE: The fourth scale anchor in head 1 passed at 99.13%, showing four coordinate anchors can work, while the first head-0 anchor at column 4 previously passed at 99.79%; this motivates testing column 4 for head-0 row 3 after its column-0 anchor failed at 71.61%. The subsequent query-bias reduction failed at 70.02%, favoring another function-preserving scale gauge.

<<<<<<< SEARCH
    """QKV projection with two zero anchors and seven query scale anchors."""
=======
    """QKV projection with two zero anchors and eight query scale anchors."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 9))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight_rest[5:12],
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
                self.weight_rest[5:12],
                fixed,
                self.weight_rest[12:23],
                fixed,
                self.weight_rest[23:26],
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
                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                scale = 0.02 / weight[4, 0]
=======
                scale = 0.02 / weight[2, 0]
                weight[2].mul_(scale)
                weight[10].div_(scale)

                scale = 0.02 / weight[3, 4]
                weight[3].mul_(scale)
                weight[11].div_(scale)

                scale = 0.02 / weight[4, 0]
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[17:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
=======
                            flat[17:28],
                            flat[29:32],
                            flat[33:40],
                            flat[41:48],
                            flat[49:56],
                            flat[57:],
>>>>>>> REPLACE