MECHANISM: Ninth-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s ninth-farthest attention bias to zero will reduce learned parameters from 1,150 to 1,148 while retaining at least 99% accuracy, because this distance affects only nine causal query-key pairs per full sequence and pruning the eight farther endpoints retained 99.87% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 10`, with fixed zeros at distance zero and the nine largest distances.

EVIDENCE: The current 1,150-parameter design achieved 99.87% accuracy after successively fixing the eight largest-distance biases, supporting the adjacent two-parameter reduction while preserving demonstrated lexical, MLP, value, and routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The eight largest distances occur in only one through eight causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 9))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The nine largest distances occur in only one through nine causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 10))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 8),
=======
                self.relative_bias.new_zeros(self.n_head, 9),
>>>>>>> REPLACE