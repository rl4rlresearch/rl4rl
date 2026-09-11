MECHANISM: Fifth-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s fifth-farthest attention bias to zero will reduce learned parameters from 1,158 to 1,156 while retaining at least 99% accuracy, because this distance affects only five causal query-key pairs per full sequence and pruning the four farther endpoints retained 99.94% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 6`, with fixed zeros at distance zero and the five largest distances.

EVIDENCE: The current 1,158-parameter design achieved 99.94% accuracy after successively fixing the four largest-distance biases, motivating the adjacent minimal reduction while preserving all demonstrated lexical, MLP, value, and routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The four largest distances occur in only one through four causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 5))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The five largest distances occur in only one through five causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 4),
=======
                self.relative_bias.new_zeros(self.n_head, 5),
>>>>>>> REPLACE