MECHANISM: Tenth-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s tenth-farthest attention bias to zero will reduce learned parameters from 1,148 to 1,146 while retaining at least 99% accuracy, because this distance affects only ten causal query-key pairs per full sequence and pruning the nine farther endpoints retained 99.96% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 11`, with fixed zeros at distance zero and the ten largest distances.

EVIDENCE: The current 1,148-parameter design achieved 99.96% accuracy after successively fixing the nine largest-distance biases, supporting the adjacent two-parameter reduction while preserving all demonstrated lexical, MLP, value, and routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The nine largest distances occur in only one through nine causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 10))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The ten largest distances occur in only one through ten causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 11))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 9),
=======
                self.relative_bias.new_zeros(self.n_head, 10),
>>>>>>> REPLACE