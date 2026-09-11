MECHANISM: Complementary endpoint bias tying

HYPOTHESIS: Tying the first head’s twelfth-farthest bias to the second head’s thirteenth-farthest bias will reduce the model from 1,117 to 1,116 parameters while maintaining at least 99% accuracy, because prior ablations show that both routes must remain present but do not establish that they require independent learned strengths.

INTENDED_EDIT: Replace the two complementary head-specific endpoint parameters with one shared scalar used at both distances.

EVIDENCE: Removing the first head’s twelfth-farthest endpoint collapsed accuracy to 6.53%, while removing the second head’s thirteenth-farthest endpoint reduced it to 73.42%; retaining their complementary routes produced 99.48%, motivating parameter tying rather than another endpoint deletion.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Retain the next
        # endpoint only for the first head and its neighbor only for the second.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 14))
        self.relative_bias_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Retain the next
        # endpoint only for the first head and its neighbor only for the second,
        # tying their complementary routing strengths to one learned scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 14))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_penultimate = torch.cat(
            (
                self.relative_bias_penultimate.new_zeros(self.n_head - 1),
                self.relative_bias_penultimate,
            )
        ).unsqueeze(1)
=======
        relative_bias_penultimate = torch.cat(
            (
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
                self.relative_bias_endpoint,
            )
        ).unsqueeze(1)
>>>>>>> REPLACE