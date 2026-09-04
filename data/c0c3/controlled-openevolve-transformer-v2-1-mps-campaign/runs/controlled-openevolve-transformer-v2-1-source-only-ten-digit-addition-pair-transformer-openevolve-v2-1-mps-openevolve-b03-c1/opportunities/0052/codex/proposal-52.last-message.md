MECHANISM: Third-farthest relative-bias pruning

HYPOTHESIS: Fixing each head’s third-farthest attention bias to zero will reduce parameters from 1,162 to 1,160 while retaining at least 99% accuracy, because this distance affects only three causal query-key pairs per full sequence and pruning the two farther endpoints retained 99.95% accuracy.

INTENDED_EDIT: Learn relative biases only for distances 1 through `max_seq_len - 4`, appending fixed zeros for distance zero and the three largest distances.

EVIDENCE: The current design achieved 99.95% accuracy after fixing the two largest-distance biases, while the immediately preceding one-endpoint design achieved 99.99%; this supports testing the adjacent next-least-used bias without reducing demonstrated MLP, lexical, or routing capacity.

<<<<<<< SEARCH
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The two largest distances occur in only one and two causal score
        # rows, so fix both sparse endpoints and learn the intervening biases.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 3))
=======
        # Distance zero fixes each head's softmax-invariant additive gauge.
        # The three largest distances occur in only one, two, and three causal
        # score rows, so fix these sparse endpoints and learn the others.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias.new_zeros(self.n_head, 2),
=======
                self.relative_bias.new_zeros(self.n_head, 3),
>>>>>>> REPLACE