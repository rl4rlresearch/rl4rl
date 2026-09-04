MECHANISM: Cross-head twenty-second-distance bias tying

HYPOTHESIS: Tying both heads’ twenty-second-farthest attention biases will reduce the model from 1,108 to 1,107 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through twenty-first-farthest ties all succeeded without removing any routing distance.

INTENDED_EDIT: Shorten the independent per-head relative-bias core by one distance and reconstruct the twenty-second-farthest bias for both heads from one shared learned scalar.

EVIDENCE: The immediately preceding twenty-first-farthest tie reduced parameters from 1,109 to 1,108 and achieved 99.96% accuracy, extending eight consecutive successful adjacent cross-head bias ties.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next eight
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 22))
        self.relative_bias_core_twenty_first = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next nine
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 23))
        self.relative_bias_core_twenty_second = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_twenty_first = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias_core_twenty_first = (
=======
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias_core_twenty_second = (
            self.relative_bias_core_twenty_second.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_twenty_first = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_twenty_first,
=======
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_twenty_second,
                relative_bias_core_twenty_first,
>>>>>>> REPLACE