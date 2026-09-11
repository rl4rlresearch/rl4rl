MECHANISM: Endpoint-skewed hierarchical boundary interpolation

HYPOTHESIS: Reconstructing head 1’s two transition biases at one-quarter and five-eighths between learned endpoints will reduce the model from 1,002 to 1,001 parameters while achieving at least 99% accuracy, because it preserves the successful midpoint hierarchy while separating the transition more strongly from the plateau than the 98.31%-accurate equal-thirds reconstruction.

INTENDED_EDIT: Remove one head-1 relative-bias parameter and derive the two preceding plateau-transition biases using quarter-step interpolation followed by a midpoint to the plateau.

EVIDENCE: Equal-thirds two-step interpolation reached 98.31%, the closest failed 1,001-parameter design, while direct plateau extension reached only 47.11%; shifting the reconstructed boundary away from the plateau tests whether insufficient transition separation caused the narrow miss.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and interpolates the preceding transition distance.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 7)),
            ]
        )
=======
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
                    [
                        self.rel_bias[1][:-1],
                        0.75 * self.rel_bias[1][-2:-1]
                        + 0.25 * self.rel_bias[1][-1:],
                        0.375 * self.rel_bias[1][-2:-1]
                        + 0.625 * self.rel_bias[1][-1:],
                        self.rel_bias[1][-1:].expand(4),
                        self.rel_bias[1].new_zeros(3),
                    ]
>>>>>>> REPLACE