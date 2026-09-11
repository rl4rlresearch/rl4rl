MECHANISM: Cross-head sharing of the eleven sparsest relative-distance biases

HYPOTHESIS: A 1,275-parameter model will retain at least 99% accuracy because the verified 1,276-parameter model achieved 99.96%, while sharing the eleventh-farthest bias affects only eleven query-key pairs at full context and preserves head-specific biases at every more frequent distance.

INTENDED_EDIT: Extend cross-head sharing from the ten to the eleven maximum-distance attention biases, reducing the model by one learned parameter.

EVIDENCE: The current 1,276-parameter design met the requirement after progressive sharing through ten sparsest distance bins; extending the same mechanism by one bin is the smallest directly supported reduction.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 11))
        self.far_rel_bias = nn.Parameter(torch.zeros(10))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
        self.far_rel_bias = nn.Parameter(torch.zeros(11))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 10),
=======
                self.far_rel_bias.expand(self.n_head, 11),
>>>>>>> REPLACE