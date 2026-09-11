MECHANISM: Farthest-distance relative-bias pruning

HYPOTHESIS: Fixing each head’s least-used maximum-distance attention bias to zero will reduce the model from 1,194 to 1,192 parameters while retaining at least 99% accuracy, because that bias affects only one causal query-key pair per full sequence.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 2` and append fixed zeros at distances zero and `max_seq_len - 1`.

EVIDENCE: The `d_ff=11` design reached 99.98%, whereas reducing `d_ff` again collapsed accuracy; this motivates preserving MLP capacity and testing a minimal reduction in the untouched 44-parameter relative-bias table.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge;
        # only relative differences between its distance biases are learned.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The maximum distance occurs in only one causal score row, so fix that
        # least-used endpoint as well and learn the intervening biases.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias = torch.cat(
            (self.relative_bias.new_zeros(self.n_head, 1), self.relative_bias),
            dim=1,
        )
=======
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
>>>>>>> REPLACE