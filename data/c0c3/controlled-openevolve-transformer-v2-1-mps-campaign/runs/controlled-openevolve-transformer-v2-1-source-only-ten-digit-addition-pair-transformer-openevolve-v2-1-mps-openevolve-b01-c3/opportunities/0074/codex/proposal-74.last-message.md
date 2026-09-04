MECHANISM: Cross-head sharing of the fourteen sparsest relative-distance attention biases

HYPOTHESIS: The 1,272-parameter model will maintain at least 99% accuracy because the identical fourteen-bin sharing design previously achieved 99.3%.

INTENDED_EDIT: Extend cross-head sharing from thirteen to fourteen maximum-distance biases, removing one learned parameter.

EVIDENCE: Reference Design 1 verified this exact 1,272-parameter mechanism at 99.3%; all tested 1,271-parameter alternatives fell below the accuracy threshold.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 14))
        self.far_rel_bias = nn.Parameter(torch.zeros(13))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 13),
=======
                self.far_rel_bias.expand(self.n_head, 14),
>>>>>>> REPLACE