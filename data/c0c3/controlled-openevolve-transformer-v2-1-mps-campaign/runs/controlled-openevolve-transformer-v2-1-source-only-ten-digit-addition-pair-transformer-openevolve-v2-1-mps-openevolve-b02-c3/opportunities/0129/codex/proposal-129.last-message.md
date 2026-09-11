MECHANISM: First-head boundary-bias suffix extension

HYPOTHESIS: A 667-parameter transformer will retain at least 99% accuracy because it preserves the qualified 668-parameter architecture while fixing only the adjacent sixth-longest first-head relative-bias coefficient.

INTENDED_EDIT: Extend the first head’s fixed long-distance relative-bias suffix from three to four coefficients, removing exactly one learned parameter.

EVIDENCE: Successive first-head boundary-bias reductions qualified at 670 parameters (99.45%), 669 parameters (99.11%), and 668 parameters (99.14%), while reductions in query bias, lexical codes, final normalization, and the remaining MLP output bias failed; continuing the only repeatedly successful reduction mechanism is the most informative next test.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head also fixes its next three longest distances.
        # The second head fixes its next six longest biases, shares its three
        # longest remaining positions, and separately shares the adjacent
        # pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 12)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The first head also fixes its next four longest distances.
        # The second head fixes its next six longest biases, shares its three
        # longest remaining positions, and separately shares the adjacent
        # pair immediately preceding them.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 13)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 3
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 3 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                self.relative_bias.new_zeros(3),
=======
        first_head_bias = self.relative_bias[
            : self.relative_bias_per_head - 4
        ]
        second_head_bias = self.relative_bias[
            self.relative_bias_per_head - 4 :
        ]
        relative_bias_coeff = torch.cat(
            (
                first_head_bias,
                self.relative_bias.new_zeros(4),
>>>>>>> REPLACE