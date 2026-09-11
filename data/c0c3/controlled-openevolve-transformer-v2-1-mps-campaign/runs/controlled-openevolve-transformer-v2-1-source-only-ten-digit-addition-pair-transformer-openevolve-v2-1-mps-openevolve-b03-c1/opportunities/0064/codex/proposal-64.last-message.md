MECHANISM: Asymmetric single-head endpoint pruning

HYPOTHESIS: Fixing only the first head’s twelfth-farthest attention bias will reduce the model from 1,120 to 1,119 learned parameters while retaining at least 99% accuracy, because the joint two-head ablation may have collapsed by removing a route needed by only one head.

INTENDED_EDIT: Split the relative biases into a shared-length per-head core and a one-parameter endpoint retained only for the second head, reconstructing the full bias table during the forward pass.

EVIDENCE: Removing both twelfth-farthest biases collapsed accuracy from 99.65% to 8.01%, while all eleven farther two-head removals succeeded; isolating one head is the smallest informative test of whether both parameters are necessary.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The eleven largest distances occur in only one through eleven causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven largest distances in both heads and the adjacent
        # twelfth-farthest distance in the first head only.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(n_head - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
=======
        endpoint = torch.cat(
            (
                self.relative_bias.new_zeros(1),
                self.relative_bias_endpoint,
            )
        ).unsqueeze(1)
        learned_relative_bias = torch.cat(
            (self.relative_bias, endpoint), dim=1
        )
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                learned_relative_bias,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
>>>>>>> REPLACE