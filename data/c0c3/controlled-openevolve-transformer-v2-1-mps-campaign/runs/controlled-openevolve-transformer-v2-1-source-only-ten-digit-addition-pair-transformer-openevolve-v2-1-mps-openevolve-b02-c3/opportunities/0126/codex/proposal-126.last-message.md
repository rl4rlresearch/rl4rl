MECHANISM: First-head boundary-distance bias fixing

HYPOTHESIS: A 670-parameter transformer will retain at least 99% accuracy because fixing the first head’s third-longest relative-bias coefficient extends its existing fixed longest-distance suffix while preserving all frequently used routing biases and the qualified 671-parameter model’s query, lexical, normalization, and MLP parameters.

INTENDED_EDIT: Remove one learned relative-bias scalar and reconstruct the first head’s third-longest distance bias as zero.

EVIDENCE: The current 671-parameter model achieved 99.22%, while further query-bias fixing failed; its two longest relative biases are already fixed for both heads, motivating an isolated extension of the first head’s sparsely used boundary-distance constraint without merging learned routing coefficients.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases, shares
        # its three longest remaining positions, and separately shares the
        # adjacent pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head also fixes its third-longest distance. The
        # second head fixes its next six longest biases, shares its three
        # longest remaining positions, and separately shares the adjacent
        # pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 10)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                second_head_bias[:-2],
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 1
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 1 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                self.relative_bias.new_zeros(1),
                second_head_bias[:-2],
>>>>>>> REPLACE