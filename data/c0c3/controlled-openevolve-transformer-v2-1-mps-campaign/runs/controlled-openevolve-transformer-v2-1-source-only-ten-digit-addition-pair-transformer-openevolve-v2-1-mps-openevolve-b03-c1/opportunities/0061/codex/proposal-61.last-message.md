MECHANISM: Eleventh-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s eleventh-farthest attention bias to zero will reduce learned parameters from 1,122 to 1,120 while retaining at least 99% accuracy, because this distance affects only eleven causal query-key pairs per full sequence.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 12`, with fixed zeros at distance zero and the eleven largest distances.

EVIDENCE: The current 1,122-parameter harmonic-position design achieved 99.51%, while ten consecutive farthest-distance pruning steps retained at least 99.77%; this supports testing the adjacent two-parameter reduction without changing the newly compressed positional representation or known load-bearing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The ten largest distances occur in only one through ten causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 11))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The eleven largest distances occur in only one through eleven causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 10),
=======
                self.relative_bias.new_zeros(self.n_head, 11),
>>>>>>> REPLACE