MECHANISM: Two-step learned head-1 boundary interpolation

HYPOTHESIS: Replacing head 1’s remaining pre-boundary bias with two evenly spaced interpolants will reduce the model from 1,002 to 1,001 parameters while maintaining at least 99% accuracy, because it preserves distinct transition values instead of extending the failed equality constraint.

INTENDED_EDIT: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct the two preceding plateau-transition biases at one-third and two-thirds between neighboring learned endpoints.

EVIDENCE: Head 1’s single learned boundary interpolation retained 99.67%, while extending the plateau by direct equality collapsed to 47.11%; this motivates lengthening the successful smooth transition without compressing the sensitive head-0 pathway.

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
        # and interpolates the preceding two transition distances.
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
                        (
                            2.0 * self.rel_bias[1][-2:-1]
                            + self.rel_bias[1][-1:]
                        )
                        / 3.0,
                        (
                            self.rel_bias[1][-2:-1]
                            + 2.0 * self.rel_bias[1][-1:]
                        )
                        / 3.0,
                        self.rel_bias[1][-1:].expand(4),
                        self.rel_bias[1].new_zeros(3),
                    ]
>>>>>>> REPLACE