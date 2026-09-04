MECHANISM: Cross-head sharing of the twelve sparsest relative-distance biases

HYPOTHESIS: A 1,274-parameter model will retain at least 99% accuracy because the verified 1,275-parameter model achieved 99.98%, while sharing the twelfth-farthest bias affects only twelve query-key pairs at full context and preserves head-specific biases at every more frequent distance.

INTENDED_EDIT: Extend cross-head sharing from the current nine to the twelve maximum-distance attention biases, reducing the current model by three learned parameters and the best verified design by one.

EVIDENCE: The 1,275-parameter design met the requirement after progressive sharing through eleven sparsest distance bins; extending the same mechanism by one bin is the smallest directly supported reduction.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 10))
        self.far_rel_bias = nn.Parameter(torch.zeros(9))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
        self.far_rel_bias = nn.Parameter(torch.zeros(12))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 9),
=======
                self.far_rel_bias.expand(self.n_head, 12),
>>>>>>> REPLACE