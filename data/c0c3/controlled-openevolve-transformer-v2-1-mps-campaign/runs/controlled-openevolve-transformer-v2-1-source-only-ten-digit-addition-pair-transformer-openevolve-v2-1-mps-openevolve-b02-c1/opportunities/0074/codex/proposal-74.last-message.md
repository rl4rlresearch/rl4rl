MECHANISM: Reference-anchored head-0 boundary interpolation

HYPOTHESIS: Interpolating head 0’s third-farthest relative bias between its learned fourth-farthest bias and the fixed zero boundary will reduce parameters from 1,002 to 1,001 while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one head-0 relative-bias parameter and reconstruct it as half of the neighboring learned boundary value, preserving rank-two content routing and all other behavior.

EVIDENCE: Directly sharing head 0’s third- and fourth-farthest biases failed at 59.01%, while learned boundary interpolation in head 1 retained 99.67%; interpolation preserves a distinct transition that equality removes.

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
        # Head 0 interpolates its third-farthest distance between the
        # preceding learned bias and its fixed far-distance reference.
        # Head 1 ties the three farthest to its reference, shares the next
        # boundary quadruplet, and interpolates the preceding transition.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 3)),
                nn.Parameter(torch.zeros(max_seq_len - 7)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 2)),
=======
                torch.cat(
                    [
                        self.rel_bias[0],
                        0.5 * self.rel_bias[0][-1:],
                        self.rel_bias[0].new_zeros(2),
                    ]
                ),
>>>>>>> REPLACE