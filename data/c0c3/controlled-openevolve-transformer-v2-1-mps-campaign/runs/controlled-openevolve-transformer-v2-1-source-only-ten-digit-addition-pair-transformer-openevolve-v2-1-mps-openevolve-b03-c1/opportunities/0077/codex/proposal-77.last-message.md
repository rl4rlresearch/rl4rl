MECHANISM: Cross-head eighteenth-distance bias tying

HYPOTHESIS: Tying both heads’ eighteenth-farthest attention biases will reduce the model from 1,112 to 1,111 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- through seventeenth-farthest ties all preserved at least 99.84% accuracy without removing any routing distance.

INTENDED_EDIT: Shorten the independent per-head relative-bias core by one distance and reconstruct the eighteenth-farthest bias for both heads from one shared learned scalar.

EVIDENCE: The immediately preceding seventeenth-farthest cross-head tie reduced parameters from 1,113 to 1,112 and achieved 99.84% accuracy, extending four consecutive successful adjacent bias ties.

<<<<<<< SEARCH
        # Fix the eleven sparsest endpoints for both heads. Share the next four
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 18))
        self.relative_bias_core_preantepenultimate = nn.Parameter(torch.zeros(1))
=======
        # Fix the eleven sparsest endpoints for both heads. Share the next five
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 19))
        self.relative_bias_core_eighteenth = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_preantepenultimate = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_preantepenultimate = (
            self.relative_bias_core_preantepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
=======
        relative_bias_core_eighteenth = (
            self.relative_bias_core_eighteenth.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_preantepenultimate = (
            self.relative_bias_core_preantepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias,
                relative_bias_core_preantepenultimate,
                relative_bias_core_antepenultimate,
=======
                self.relative_bias,
                relative_bias_core_eighteenth,
                relative_bias_core_preantepenultimate,
                relative_bias_core_antepenultimate,
>>>>>>> REPLACE