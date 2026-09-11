MECHANISM: Cross-head fourteenth-distance bias tying

HYPOTHESIS: Tying both heads’ fourteenth-farthest attention biases will reduce the model from 1,116 to 1,115 parameters while retaining at least 99% accuracy, because complementary endpoint tying reached 99.87%, showing that distinct routing paths can share a learned strength without losing accuracy.

INTENDED_EDIT: Remove the final column from each head’s relative-bias core and reconstruct that distance for both heads from one shared learned scalar.

EVIDENCE: The current design tied two load-bearing, head-specific endpoint biases and achieved 99.87%; this supports testing the smallest adjacent sharing change while preserving every routing distance.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Retain the next
        # endpoint only for the first head and its neighbor only for the second,
        # tying their complementary routing strengths to one learned scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 14))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next
        # learned distance across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_penultimate = torch.cat(
            (
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
                self.relative_bias_endpoint,
            )
        ).unsqueeze(1)
        relative_bias_endpoint = torch.cat(
            (
                self.relative_bias_endpoint,
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
            )
        ).unsqueeze(1)
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_penultimate,
                relative_bias_endpoint,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
=======
        relative_bias_core_endpoint = self.relative_bias_core_endpoint.expand(
            self.n_head
        ).unsqueeze(1)
        relative_bias_penultimate = torch.cat(
            (
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
                self.relative_bias_endpoint,
            )
        ).unsqueeze(1)
        relative_bias_endpoint = torch.cat(
            (
                self.relative_bias_endpoint,
                self.relative_bias_endpoint.new_zeros(self.n_head - 1),
            )
        ).unsqueeze(1)
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_endpoint,
                relative_bias_penultimate,
                relative_bias_endpoint,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
>>>>>>> REPLACE