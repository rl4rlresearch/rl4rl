MECHANISM: Cross-head fifteenth-distance bias tying

HYPOTHESIS: Tying both heads’ fifteenth-farthest attention biases will reduce the model from 1,115 to 1,114 parameters while retaining at least 99% accuracy, because the adjacent fourteenth-farthest cross-head tie achieved 99.92% without removing any routing distance.

INTENDED_EDIT: Remove the final column from each head’s relative-bias core and reconstruct that fifteenth-farthest distance from one new shared learned scalar.

EVIDENCE: Cross-head tying at the immediately adjacent fourteenth-farthest distance reduced parameters from 1,116 to 1,115 and improved verified accuracy to 99.92%, motivating the same one-parameter sharing test while preserving all attention distances.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next
        # learned distance across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next two
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 16))
        self.relative_bias_core_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_endpoint = self.relative_bias_core_endpoint.expand(
            self.n_head
        ).unsqueeze(1)
        relative_bias_penultimate = torch.cat(
=======
        relative_bias_core_penultimate = (
            self.relative_bias_core_penultimate.expand(self.n_head).unsqueeze(1)
        )
        relative_bias_core_endpoint = self.relative_bias_core_endpoint.expand(
            self.n_head
        ).unsqueeze(1)
        relative_bias_penultimate = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_endpoint,
                relative_bias_penultimate,
=======
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_penultimate,
                relative_bias_core_endpoint,
                relative_bias_penultimate,
>>>>>>> REPLACE