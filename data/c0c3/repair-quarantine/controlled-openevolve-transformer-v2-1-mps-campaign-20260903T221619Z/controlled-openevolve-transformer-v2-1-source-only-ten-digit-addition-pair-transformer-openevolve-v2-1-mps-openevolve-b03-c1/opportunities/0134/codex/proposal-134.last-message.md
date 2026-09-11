MECHANISM: Tenth cross-head far-distance bias tie

HYPOTHESIS: Sharing the next adjacent relative-attention bias across both heads will reduce the model from 606 to 605 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Shorten the head-specific relative-bias table by one column and reconstruct the removed column as a zero-initialized learned scalar shared across both heads.

EVIDENCE: The verified 606-parameter model achieved 99.89% accuracy while already sharing the nine immediately farther bias distances; extending the same constraint to the adjacent distance is the smallest evidence-supported change, and earlier attempts were unverifiable rather than accuracy failures.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next nine
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 23))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next ten
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 24))
        self.relative_bias_core_twenty_third = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias_core_twenty_second = (
=======
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias_core_twenty_third = (
            self.relative_bias_core_twenty_third.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_twenty_second = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_twenty_second,
=======
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_twenty_third,
                relative_bias_core_twenty_second,
>>>>>>> REPLACE