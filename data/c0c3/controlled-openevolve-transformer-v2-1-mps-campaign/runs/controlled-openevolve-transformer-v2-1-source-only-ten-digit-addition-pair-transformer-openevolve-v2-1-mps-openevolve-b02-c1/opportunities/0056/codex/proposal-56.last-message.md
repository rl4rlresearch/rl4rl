MECHANISM: Per-head three-distance boundary tying

HYPOTHESIS: Tying each head’s three farthest relative-distance biases will reduce the model from 1,044 to 1,042 parameters while retaining at least 99% accuracy, because tying the two farthest distances already achieved 99.79% and this changes only one additional sparse distance per head.

INTENDED_EDIT: Store one fewer relative-bias coordinate per head and reconstruct the three farthest distances from each head’s existing common-mode reference.

EVIDENCE: The current two-distance tie retained 99.79% accuracy at 1,044 parameters, while aggressive affine compression failed completely; the smallest further positional-table ablation is therefore one additional boundary tie per head.

<<<<<<< SEARCH
        # The two farthest distances share each head's softmax-invisible
        # common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
=======
        # The three farthest distances share each head's softmax-invisible
        # common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        rel_bias = F.pad(self.rel_bias, (0, 2))
=======
        rel_bias = F.pad(self.rel_bias, (0, 3))
>>>>>>> REPLACE