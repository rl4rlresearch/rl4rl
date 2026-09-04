MECHANISM: Cross-head adjacent-boundary bias sharing

HYPOTHESIS: Sharing head 0’s third- and fourth-farthest learned relative biases will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because it preserves an adaptive boundary value instead of extending head 1’s failed quintuplet or tying head 0’s third-farthest bias to the fixed reference.

INTENDED_EDIT: Shorten head 0’s relative-bias parameter by one coordinate and reconstruct its third-farthest bias from the neighboring fourth-farthest learned value; leave head 1’s successful quadruplet sharing and all other behavior unchanged.

EVIDENCE: Head 1’s learned adjacent-boundary sharing retained 99.68% through four biases, whereas extending it to five collapsed to 47.11% and direct reference tying previously collapsed to 14.36%; this motivates testing the same adaptive adjacent-sharing mechanism in head 0 without further compressing head 1.

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
        # Head 0 ties the two farthest distances and shares the preceding
        # learned boundary pair. Head 1 ties the three farthest to its
        # reference and shares the next boundary quadruplet.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 3)),
                nn.Parameter(torch.zeros(max_seq_len - 6)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 2)),
=======
                torch.cat(
                    [
                        self.rel_bias[0],
                        self.rel_bias[0][-1:],
                        self.rel_bias[0].new_zeros(2),
                    ]
                ),
>>>>>>> REPLACE