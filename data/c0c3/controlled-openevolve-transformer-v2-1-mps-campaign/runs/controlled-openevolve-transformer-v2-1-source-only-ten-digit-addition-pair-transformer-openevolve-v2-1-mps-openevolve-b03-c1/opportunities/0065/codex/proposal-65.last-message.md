MECHANISM: Complementary single-head endpoint pruning

HYPOTHESIS: Fixing only the second head’s twelfth-farthest attention bias will reduce the model from 1,120 to 1,119 learned parameters while retaining at least 99% accuracy, because the failed complementary ablation indicates that the first head may be the sole head requiring this routing endpoint.

INTENDED_EDIT: Split the relative biases into a per-head core and a one-parameter twelfth-farthest endpoint retained only for the first head, reconstructing the full bias table during the forward pass.

EVIDENCE: Pruning both twelfth-farthest biases collapsed accuracy to 8.01%, and pruning only the first head’s endpoint collapsed it to 6.53%; the untested complementary ablation directly determines whether the first head’s endpoint is load-bearing while the second head’s parameter is redundant.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The eleven largest distances occur in only one through eleven causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads and the adjacent
        # endpoint for the second head, retaining it only for the first head.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
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
                relative_bias_endpoint,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
>>>>>>> REPLACE