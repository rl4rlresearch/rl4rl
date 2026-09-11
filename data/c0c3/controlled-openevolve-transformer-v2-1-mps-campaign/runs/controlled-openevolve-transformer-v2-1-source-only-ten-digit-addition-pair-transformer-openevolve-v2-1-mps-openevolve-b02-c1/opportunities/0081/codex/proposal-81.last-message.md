MECHANISM: Successor-preserving head-1 transition interpolation

HYPOTHESIS: Reconstructing the removed head-1 boundary bias at one-eighth between learned endpoints will reduce the model from 1,001 to 1,000 parameters while retaining at least 99% accuracy, because the two downstream transition fractions from the 99.97%-accurate design remain unchanged.

INTENDED_EDIT: Shorten head 1’s relative-bias parameter by one coordinate, insert a conservative one-eighth interpolant for the removed boundary, and preserve the successful quarter and five-eighths transition biases.

EVIDENCE: The current quarter/five-eighths reconstruction achieved 99.97%, whereas the 1,000-parameter recursive reconstruction achieved 97.32% after changing all downstream transition fractions; preserving those successful fractions isolates compression of only the additional boundary.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the three preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 9)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                        self.rel_bias[1][:-1],
                        0.75 * self.rel_bias[1][-2:-1]
                        + 0.25 * self.rel_bias[1][-1:],
                        0.375 * self.rel_bias[1][-2:-1]
                        + 0.625 * self.rel_bias[1][-1:],
                        self.rel_bias[1][-1:].expand(4),
=======
                        self.rel_bias[1][:-1],
                        0.875 * self.rel_bias[1][-2:-1]
                        + 0.125 * self.rel_bias[1][-1:],
                        0.75 * self.rel_bias[1][-2:-1]
                        + 0.25 * self.rel_bias[1][-1:],
                        0.375 * self.rel_bias[1][-2:-1]
                        + 0.625 * self.rel_bias[1][-1:],
                        self.rel_bias[1][-1:].expand(4),
>>>>>>> REPLACE