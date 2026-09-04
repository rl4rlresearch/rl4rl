MECHANISM: Cross-head twenty-third-distance bias tying

HYPOTHESIS: Replacing the two head-specific biases at the twenty-third-farthest attention distance with one shared learned scalar will reduce the model from 1,107 to 1,106 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Shorten the independent per-head relative-bias core by one column, add an explicitly used shared scalar for the removed distance, and prepend it to the existing shared-distance sequence.

EVIDENCE: Sharing every adjacent distance from the fourteenth- through twenty-second-farthest retained at least 99% accuracy, with the twenty-second tie reaching 99.91% at 1,107 parameters; prior attempts at this next tie did not produce adverse accuracy evidence because they reproduced the existing implementation or were unverifiable.

<<<<<<< SEARCH
        # Fix the eleven sparsest endpoints for both heads. Share the next nine
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 23))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
=======
        # Fix the eleven sparsest endpoints for both heads. Share the next ten
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
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
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_twenty_second,
                relative_bias_core_twenty_first,
=======
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_twenty_third,
                relative_bias_core_twenty_second,
                relative_bias_core_twenty_first,
>>>>>>> REPLACE