MECHANISM: Complementary head-specific thirteenth-distance routing

HYPOTHESIS: Retaining the thirteenth-farthest bias only for the second head will reduce the model from 1,118 to 1,117 parameters while maintaining at least 99% accuracy, because removing that second-head bias caused the partial collapse to 73.42%, whereas the preceding distance showed that only one head needed its endpoint.

INTENDED_EDIT: Shorten the shared per-head relative-bias core by one distance, retain the twelfth-farthest endpoint only for the first head, and retain the thirteenth-farthest endpoint only for the second head.

EVIDENCE: Pruning the second head’s thirteenth-farthest bias reduced accuracy to 73.42%, directly indicating that this endpoint should be restored; the complementary first-head pruning remains untested and mirrors the successful asymmetric pruning at the adjacent distance.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads and the adjacent
        # endpoint for the second head, retaining it only for the first head.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # Fix the eleven sparsest endpoints for both heads. Retain the next
        # endpoint only for the first head and its neighbor only for the second.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 14))
        self.relative_bias_penultimate = nn.Parameter(torch.zeros(1))
        self.relative_bias_endpoint = nn.Parameter(torch.zeros(1))
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
        relative_bias_penultimate = torch.cat(
            (
                self.relative_bias_penultimate.new_zeros(self.n_head - 1),
                self.relative_bias_penultimate,
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
>>>>>>> REPLACE