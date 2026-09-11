MECHANISM: Merge head 1’s terminal relative-bias routing knots

HYPOTHESIS: Tying head 1’s two terminal learned routing knots into one seven-distance band will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy, because head 1 already tolerates broader far-distance sharing and this preserves a learned nonzero band before its three reference-fixed distances.

INTENDED_EDIT: Store one fewer head-1 relative-bias parameter and replace its two interpolated transition distances plus four-distance plateau with a single learned seven-distance plateau.

EVIDENCE: Head 0 retained 99.89% accuracy after progressively tying five farthest distances, while head 1 already uses stronger structured far-distance sharing; merging its adjacent terminal knots is an isolated one-parameter lookup compression, unlike the failed replacement of both lookup tables with Gaussian routing bands.

<<<<<<< SEARCH
        # Head 0 ties the five farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 5)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 ties the five farthest distances. Head 1 ties the three
        # farthest to its reference and shares the preceding seven-distance
        # routing band.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 5)),
                nn.Parameter(torch.zeros(max_seq_len - 9)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                torch.cat(
                    [
                        self.rel_bias[1][:-1],
                        0.75 * self.rel_bias[1][-2:-1]
                        + 0.25 * self.rel_bias[1][-1:],
                        0.375 * self.rel_bias[1][-2:-1]
                        + 0.625 * self.rel_bias[1][-1:],
                        self.rel_bias[1][-1:].expand(4),
                        self.rel_bias[1].new_zeros(3),
                    ]
                ),
=======
                torch.cat(
                    [
                        self.rel_bias[1][:-1],
                        self.rel_bias[1][-1:].expand(7),
                        self.rel_bias[1].new_zeros(3),
                    ]
                ),
>>>>>>> REPLACE