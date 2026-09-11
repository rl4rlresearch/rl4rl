MECHANISM: Twelfth-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s twelfth-farthest attention bias to zero will reduce learned parameters from 1,120 to 1,118 while retaining at least 99% accuracy, because this distance affects only twelve causal query-key pairs per full sequence.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 13`, with fixed zeros at distance zero and the twelve largest distances.

EVIDENCE: The current 1,120-parameter design achieved 99.65% after eleven consecutive farthest-distance biases were fixed, motivating the adjacent two-parameter reduction while preserving all demonstrated lexical, positional, MLP, value, and routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The eleven largest distances occur in only one through eleven causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The twelve largest distances occur in only one through twelve causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 11),
=======
                self.relative_bias.new_zeros(self.n_head, 12),
>>>>>>> REPLACE