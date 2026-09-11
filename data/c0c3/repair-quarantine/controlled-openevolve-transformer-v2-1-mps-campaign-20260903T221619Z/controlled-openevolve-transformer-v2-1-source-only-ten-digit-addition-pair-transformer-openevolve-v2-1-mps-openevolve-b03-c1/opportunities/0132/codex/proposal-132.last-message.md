MECHANISM: Tenth cross-head far-distance bias tie

HYPOTHESIS: Sharing the twenty-third-farthest relative-attention bias across both heads will reduce the verified 606-parameter model to 605 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Shorten the head-specific relative-bias table by one column and reconstruct the removed column from one zero-initialized learned scalar shared across both heads.

EVIDENCE: The 606-parameter model achieved 99.89% accuracy, while sharing the nine adjacent farther-distance biases already retained at least 99%; prior attempts at this boundary were unverifiable rather than accuracy failures.

<<<<<<< SEARCH
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 23))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
=======
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 24))
        self.relative_bias_core_twenty_third = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_twenty_second = (
            self.relative_bias_core_twenty_second.expand(
                self.n_head
            ).unsqueeze(1)
        )
=======
        relative_bias_core_twenty_third = (
            self.relative_bias_core_twenty_third.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_twenty_second = (
            self.relative_bias_core_twenty_second.expand(
                self.n_head
            ).unsqueeze(1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias,
                relative_bias_core_twenty_second,
=======
                self.relative_bias,
                relative_bias_core_twenty_third,
                relative_bias_core_twenty_second,
>>>>>>> REPLACE