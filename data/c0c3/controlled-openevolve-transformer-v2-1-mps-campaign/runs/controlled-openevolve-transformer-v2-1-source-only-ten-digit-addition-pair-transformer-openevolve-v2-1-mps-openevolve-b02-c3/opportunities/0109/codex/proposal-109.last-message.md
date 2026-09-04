MECHANISM: Four-way tolerant-head relative-bias sharing

HYPOTHESIS: A 676-parameter transformer will retain at least 99% accuracy because extending the second head’s successful three-way bias sharing to four adjacent long-distance positions preserves a learned routing coefficient while leaving the first head and all normalization scales unchanged.

INTENDED_EDIT: Remove one relative-bias parameter and reuse the second head’s final learned coefficient across its four longest remaining bias positions.

EVIDENCE: The 677-parameter three-way sharing design achieved 99.97%, matching the 678-parameter two-way sharing design’s 99.97%; this directly motivates the next one-parameter extension of the same sharing pattern.

<<<<<<< SEARCH
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its three longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 8)
        )
=======
        # Distance zero and the two longest distances are fixed for both
        # heads. The second head fixes its next six longest biases and shares
        # one coefficient across its four longest remaining positions.
        self.relative_bias_per_head = max_seq_len - 3
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * self.relative_bias_per_head - 9)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                first_head_bias,
                second_head_bias,
                second_head_bias[-1:].expand(2),
                self.relative_bias.new_zeros(6),
=======
                first_head_bias,
                second_head_bias,
                second_head_bias[-1:].expand(3),
                self.relative_bias.new_zeros(6),
>>>>>>> REPLACE