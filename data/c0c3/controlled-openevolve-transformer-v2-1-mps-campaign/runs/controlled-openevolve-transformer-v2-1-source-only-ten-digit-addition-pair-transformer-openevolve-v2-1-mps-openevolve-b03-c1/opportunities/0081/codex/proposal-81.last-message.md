MECHANISM: Cross-head twenty-first-distance bias tying

HYPOTHESIS: Tying both heads’ twenty-first-farthest attention biases will reduce the model from 1,109 to 1,108 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through twentieth-farthest ties all succeeded without removing any routing distance.

INTENDED_EDIT: Shorten the independent per-head relative-bias core by one distance and reconstruct the twenty-first-farthest bias for both heads from one shared learned scalar.

EVIDENCE: The immediately preceding twentieth-farthest cross-head tie reduced parameters from 1,110 to 1,109 and achieved 99.93% accuracy, extending seven consecutive successful adjacent bias ties; the failed direct bilinear refactor shows that preserving the existing attention score structure is safer than compressing its query/key parameterization.

<<<<<<< SEARCH
        # Fix the eleven sparsest endpoints for both heads. Share the next seven
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 21))
        self.relative_bias_core_twentieth = nn.Parameter(torch.zeros(1))
=======
        # Fix the eleven sparsest endpoints for both heads. Share the next eight
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 22))
        self.relative_bias_core_twenty_first = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_twentieth = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_twentieth = (
            self.relative_bias_core_twentieth.expand(
                self.n_head
            ).unsqueeze(1)
        )
=======
        relative_bias_core_twenty_first = (
            self.relative_bias_core_twenty_first.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_twentieth = (
            self.relative_bias_core_twentieth.expand(
                self.n_head
            ).unsqueeze(1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias,
                relative_bias_core_twentieth,
                relative_bias_core_nineteenth,
=======
                self.relative_bias,
                relative_bias_core_twenty_first,
                relative_bias_core_twentieth,
                relative_bias_core_nineteenth,
>>>>>>> REPLACE