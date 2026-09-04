MECHANISM: Fixed extreme-distance attention bias

HYPOTHESIS: A 1,269-parameter model will achieve at least 99% accuracy because fixing only the farthest relative-attention bias at its zero initialization preserves all more frequently used learned bins and avoids coupling gradients into the penultimate bin.

INTENDED_EDIT: Replace the learned farthest-distance scalar with a fixed zero while retaining thirteen independently learned far-distance biases.

EVIDENCE: The 1,270-parameter model achieved 99.31%, while tying the two extreme-distance bins reached 97.21%; fixing the single farthest bin isolates the reduction to one query-key pair per head instead of constraining and perturbing the more frequently used penultimate bias.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(13))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 14),
=======
                self.rel_bias,
                F.pad(self.far_rel_bias, (0, 1)).expand(self.n_head, 14),
>>>>>>> REPLACE