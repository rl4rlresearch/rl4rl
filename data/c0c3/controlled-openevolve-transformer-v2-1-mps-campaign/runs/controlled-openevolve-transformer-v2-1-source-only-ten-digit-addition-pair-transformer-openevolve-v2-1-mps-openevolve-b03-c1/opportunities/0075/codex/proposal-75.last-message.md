MECHANISM: Cross-head sixteenth-distance bias tying

HYPOTHESIS: Tying both heads’ sixteenth-farthest attention biases will reduce the model from 1,114 to 1,113 parameters while retaining at least 99% accuracy, because the adjacent fourteenth- and fifteenth-farthest cross-head ties achieved 99.92% and 99.97% without removing any routing distance.

INTENDED_EDIT: Shorten the independent per-head relative-bias core by one distance and reconstruct the sixteenth-farthest bias for both heads from one shared learned scalar.

EVIDENCE: Cross-head tying at the fifteenth-farthest distance reduced parameters from 1,115 to 1,114 while improving verified accuracy to 99.97%; the immediately adjacent sixteenth-farthest distance is the smallest informative continuation of that successful compression pattern.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next two
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 16))
        self.relative_bias_core_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Share the next three
        # learned distances across heads, then retain the two complementary
        # head-specific endpoints with one additional shared scalar.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 17))
        self.relative_bias_core_antepenultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_core_endpoint = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias_core_penultimate = (
            self.relative_bias_core_penultimate.expand(self.n_head).unsqueeze(1)
        )
        relative_bias_core_endpoint = self.relative_bias_core_endpoint.expand(
            self.n_head
        ).unsqueeze(1)
=======
        relative_bias_core_antepenultimate = (
            self.relative_bias_core_antepenultimate.expand(
                self.n_head
            ).unsqueeze(1)
        )
        relative_bias_core_penultimate = (
            self.relative_bias_core_penultimate.expand(self.n_head).unsqueeze(1)
        )
        relative_bias_core_endpoint = self.relative_bias_core_endpoint.expand(
            self.n_head
        ).unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_penultimate,
                relative_bias_core_endpoint,
=======
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_core_antepenultimate,
                relative_bias_core_penultimate,
                relative_bias_core_endpoint,
>>>>>>> REPLACE