MECHANISM: Learned relative-boundary interpolation

HYPOTHESIS: Reconstructing head 1’s fifth boundary bias as the midpoint of its learned neighbors will reduce the model from 1,039 to 1,038 parameters while retaining at least 99% accuracy, because it preserves a distinct transition value that direct quintuplet sharing eliminated.

INTENDED_EDIT: Shorten head 1’s relative-bias parameter by one coordinate and interpolate the removed pre-boundary bias between the preceding independent bias and the successful shared quadruplet.

EVIDENCE: Head 1’s adaptive boundary quadruplet achieved 99.68%, while extending equality to a quintuplet collapsed to 47.11%; this indicates the removed transition needs distinction, motivating learned interpolation rather than equality.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary quadruplet.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 6)),
            ]
        )
=======
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and interpolates the preceding transition distance.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 7)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                torch.cat(
                    [
                        self.rel_bias[1],
                        self.rel_bias[1][-1:].expand(3),
                        self.rel_bias[1].new_zeros(3),
                    ]
                ),
=======
                torch.cat(
                    [
                        self.rel_bias[1][:-1],
                        0.5
                        * (
                            self.rel_bias[1][-2:-1]
                            + self.rel_bias[1][-1:]
                        ),
                        self.rel_bias[1][-1:].expand(4),
                        self.rel_bias[1].new_zeros(3),
                    ]
                ),
>>>>>>> REPLACE