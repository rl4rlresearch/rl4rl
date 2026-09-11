MECHANISM: Seventh-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s seventh-farthest attention bias to zero will reduce learned parameters from 1,154 to 1,152 while retaining at least 99% accuracy, because this distance affects only seven causal query-key pairs per full sequence and pruning the six farther endpoints retained 99.92% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 8`, with fixed zeros at distance zero and the seven largest distances.

EVIDENCE: The current 1,154-parameter design achieved 99.92% accuracy after successively fixing the six largest-distance biases, supporting the adjacent two-parameter reduction while preserving demonstrated lexical, MLP, value, and routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The six largest distances occur in only one through six causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 7))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The seven largest distances occur in only one through seven causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 8))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 6),
=======
                self.relative_bias.new_zeros(self.n_head, 7),
>>>>>>> REPLACE