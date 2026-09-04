MECHANISM: Cascaded asymmetric endpoint pruning

HYPOTHESIS: Fixing the second head’s thirteenth-farthest attention bias will reduce the model from 1,119 to 1,118 learned parameters while retaining at least 99% accuracy, because that head already retained 99.53% accuracy without its twelfth-farthest endpoint while the complementary first-head ablation collapsed.

INTENDED_EDIT: Shorten the per-head relative-bias core by one distance and retain the two adjacent endpoints only for the first head.

EVIDENCE: Pruning the second head’s twelfth-farthest endpoint achieved 99.53%, whereas pruning the first head’s corresponding endpoint achieved only 6.53%; this head-specific contrast motivates extending pruning by one adjacent distance only on the apparently less endpoint-dependent second head.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads and the adjacent
        # endpoint for the second head, retaining it only for the first head.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads and the next two
        # endpoints for the second head, retaining both only for the first.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 14))
        self.relative_bias_endpoints = nn.Parameter(torch.zeros(2))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        relative_bias_endpoints = torch.cat(
            (
                self.relative_bias_endpoints.unsqueeze(1),
                self.relative_bias_endpoints.new_zeros(2, self.n_head - 1),
            ),
            dim=1,
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                relative_bias_endpoints,
                self.relative_bias.new_zeros(self.n_head, 11),
            ),
            dim=1,
        )
>>>>>>> REPLACE