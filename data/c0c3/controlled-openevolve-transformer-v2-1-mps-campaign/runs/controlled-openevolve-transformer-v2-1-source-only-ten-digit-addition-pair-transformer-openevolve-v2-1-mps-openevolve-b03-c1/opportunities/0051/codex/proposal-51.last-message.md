MECHANISM: Second-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s second-farthest attention bias to zero will reduce parameters from 1,164 to 1,162 while retaining at least 99% accuracy, because this distance affects only two causal query-key pairs per full sequence and pruning the even-farthest endpoint retained 99.99% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 3`, appending fixed zeros for distance zero and the two largest distances.

EVIDENCE: The 1,192-parameter design retained 99.99% accuracy after fixing the maximum-distance bias, and shared values subsequently retained 99.98%; this supports testing the adjacent, next-least-used bias while preserving all demonstrated routing and MLP capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The maximum distance occurs in only one causal score row, so fix that
        # least-used endpoint as well and learn the intervening biases.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The two largest distances occur in only one and two causal score
        # rows, so fix both sparse endpoints and learn the intervening biases.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
=======
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 2),
            ),
            dim=1,
        )
>>>>>>> REPLACE