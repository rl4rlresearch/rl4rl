MECHANISM: Eighth-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s eighth-farthest attention bias to zero will reduce learned parameters from 1,152 to 1,150 while retaining at least 99% accuracy, because this distance affects only eight causal query-key pairs per full sequence and pruning the seven farther endpoints retained 99.77% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 9`, with fixed zeros at distance zero and the eight largest distances.

EVIDENCE: The current 1,152-parameter design achieved 99.77% accuracy after successively fixing the seven largest-distance biases, supporting the adjacent two-parameter reduction while preserving demonstrated lexical, MLP, value, and routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The seven largest distances occur in only one through seven causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 8))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The eight largest distances occur in only one through eight causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 9))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 7),
=======
                self.relative_bias.new_zeros(self.n_head, 8),
>>>>>>> REPLACE