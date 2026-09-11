MECHANISM: Adjacent head-0 terminal-knot sharing

HYPOTHESIS: Sharing head 0’s two farthest learned relative-bias coefficients will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy, because five farther distances already share the zero reference while preserving 99.89% accuracy, and this keeps that verified routing cutoff unchanged.

INTENDED_EDIT: Store one fewer head-0 relative-bias parameter and reuse its final learned coefficient for the adjacent terminal distance, leaving the five reference-fixed distances unchanged.

EVIDENCE: Head 0 achieved 99.89% accuracy with its five farthest distances tied to zero; unlike the repeatedly timed-out six-distance extension, this tests a distinct one-parameter compression while preserving the verified five-distance zero region.

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
        # Head 0 ties its two terminal learned distances and fixes the five
        # farther distances to its reference. Head 1 ties the three farthest
        # to its reference, shares the next boundary quadruplet, and
        # reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 6)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 5)),
=======
                torch.cat(
                    [
                        self.rel_bias[0],
                        self.rel_bias[0][-1:],
                        self.rel_bias[0].new_zeros(5),
                    ]
                ),
>>>>>>> REPLACE