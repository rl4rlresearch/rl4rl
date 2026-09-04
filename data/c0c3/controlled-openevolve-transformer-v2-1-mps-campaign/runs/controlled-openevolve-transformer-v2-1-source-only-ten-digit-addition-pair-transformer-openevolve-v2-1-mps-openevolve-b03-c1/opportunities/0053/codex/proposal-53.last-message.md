MECHANISM: Fourth-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s fourth-farthest attention bias to zero will reduce parameters from 1,160 to 1,158 while retaining at least 99% accuracy, because this distance affects only four causal query-key pairs per full sequence and pruning the three farther endpoints retained 99.94% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 5`, appending fixed zeros for distance zero and the four largest distances.

EVIDENCE: The current 1,160-parameter design achieved 99.94% accuracy after successively fixing the three largest-distance biases; this tests the adjacent next-least-used bias while preserving all demonstrated lexical, MLP, value, and routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The three largest distances occur in only one, two, and three causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 4))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The four largest distances occur in only one through four causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 3),
=======
                self.relative_bias.new_zeros(self.n_head, 4),
>>>>>>> REPLACE