MECHANISM: Extend the verified query/key shear gauge to query row 2

HYPOTHESIS: Fixing `qkv.weight[2,1]` at zero through query row 3’s `0.02` pivot, with the compensating key-row shear, will reduce the model from 1562 to 1561 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add one zero anchor to the QKV parameterization, initialize it with an attention-score-preserving shear, and omit that scalar from learned storage.

EVIDENCE: The current 1562-parameter design achieved 0.9999 accuracy after applying the identical row-3-pivot construction to `qkv.weight[1,1]`; row 3 also preserves row 2’s existing component-zero and component-four anchors.

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 17))
=======
        self.weight_rest = nn.Parameter(torch.empty(out_features * in_features - 18))
>>>>>>> REPLACE

<<<<<<< SEARCH
                fixed,
                self.weight_rest[10:13],
                zero,
                self.weight_rest[13:16],
                zero,
                fixed,
                self.weight_rest[16:18],
                zero,
                self.weight_rest[18:21],
                fixed,
                self.weight_rest[21:28],
                fixed,
                self.weight_rest[28:35],
                fixed,
                self.weight_rest[35:42],
                fixed,
                self.weight_rest[42:52],
                zero,
                self.weight_rest[52:83],
                zero,
                self.weight_rest[83:],
=======
                fixed,
                zero,
                self.weight_rest[10:12],
                zero,
                self.weight_rest[12:15],
                zero,
                fixed,
                self.weight_rest[15:17],
                zero,
                self.weight_rest[17:20],
                fixed,
                self.weight_rest[20:27],
                fixed,
                self.weight_rest[27:34],
                fixed,
                self.weight_rest[34:41],
                fixed,
                self.weight_rest[41:51],
                zero,
                self.weight_rest[51:82],
                zero,
                self.weight_rest[82:],
>>>>>>> REPLACE

<<<<<<< SEARCH
                shear = weight[1, 1] / weight[3, 1]
                weight[1].sub_(shear * weight[3])
                weight[11].add_(shear * weight[9])
                weight[1, 1] = 0.0

                flat = weight.flatten()
=======
                shear = weight[1, 1] / weight[3, 1]
                weight[1].sub_(shear * weight[3])
                weight[11].add_(shear * weight[9])
                weight[1, 1] = 0.0

                # Reuse the same stable pivot to eliminate row 2's matching
                # coordinate. Row 3's zero anchors preserve row 2's existing
                # component-zero and component-four anchors.
                shear = weight[2, 1] / weight[3, 1]
                weight[2].sub_(shear * weight[3])
                weight[11].add_(shear * weight[10])
                weight[2, 1] = 0.0

                flat = weight.flatten()
>>>>>>> REPLACE

<<<<<<< SEARCH
                            flat[13:16],
                            flat[17:20],
                            flat[21:24],
=======
                            flat[13:16],
                            flat[18:20],
                            flat[21:24],
>>>>>>> REPLACE