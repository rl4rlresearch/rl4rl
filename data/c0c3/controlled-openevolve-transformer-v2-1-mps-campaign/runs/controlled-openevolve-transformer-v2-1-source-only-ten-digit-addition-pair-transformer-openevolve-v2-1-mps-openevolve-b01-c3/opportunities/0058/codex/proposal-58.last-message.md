MECHANISM: Cross-head sharing of the seven sparsest relative-distance biases

HYPOTHESIS: A 1,279-parameter model will retain at least 99% accuracy because the verified 1,280-parameter model achieved 100%, while sharing the seventh-farthest bias affects only seven query-key pairs at full context and preserves head-specific biases at every more frequent distance.

INTENDED_EDIT: Extend cross-head sharing from the six to the seven maximum-distance attention biases, reducing the model by one learned parameter beyond the verified 1,280-parameter design.

EVIDENCE: Progressive sharing of one through six sparsest distance bins consistently exceeded 99% accuracy, and the six-bin 1,280-parameter design achieved 100%; extending the same mechanism by one bin is the smallest directly supported reduction.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 5))
        self.far_rel_bias = nn.Parameter(torch.zeros(4))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 8))
        self.far_rel_bias = nn.Parameter(torch.zeros(7))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 4),
=======
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 7),
>>>>>>> REPLACE