MECHANISM: Cross-head seventeenth-distance bias tying

HYPOTHESIS: Tying both heads’ seventeenth-farthest attention biases will reduce the model from 1,113 to 1,112 parameters while retaining at least 99% accuracy, because the adjacent fourteenth-, fifteenth-, and sixteenth-farthest ties all preserved at least 99.92% accuracy without removing any routing distance.

INTENDED_EDIT: Shorten the independent per-head relative-bias core by one distance and reconstruct the seventeenth-farthest bias for both heads from one shared learned scalar.

EVIDENCE: The immediately preceding sixteenth-farthest cross-head tie reduced parameters from 1,114 to 1,113 and achieved 99.93% accuracy, extending a three-step sequence of successful adjacent ties.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next three
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 17))
        self.relative_bias_core_antepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next four
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 18))
        self.relative_bias_core_preantepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_antepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_antepenultimate = (
            self.relative_bias_core_antepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_penultimate = (
=======
        relative_bias_core_preantepenultimate = (
            self.relative_bias_core_preantepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_antepenultimate = (
            self.relative_bias_core_antepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_penultimate = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_antepenultimate,
                relative_bias_core_penultimate,
=======
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_preantepenultimate,
                relative_bias_core_antepenultimate,
                relative_bias_core_penultimate,
>>>>>>> REPLACE