MECHANISM: Cross-head sharing of the ten sparsest relative-distance biases

HYPOTHESIS: A 1,276-parameter model will retain at least 99% accuracy because the verified 1,277-parameter model achieved 100%, while sharing the tenth-farthest bias affects only ten query-key pairs at full context and preserves head-specific biases at every more frequent distance.

INTENDED_EDIT: Extend cross-head sharing from the current two to the ten maximum-distance attention biases, reducing the current model by eight learned parameters and the best verified design by one.

EVIDENCE: Progressive sharing through nine sparsest distance bins consistently exceeded 99% accuracy, and the nine-bin 1,277-parameter design achieved 100%; sharing the next-sparsest bin is the smallest directly supported reduction.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 3))
        self.far_rel_bias = nn.Parameter(torch.zeros(2))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 11))
        self.far_rel_bias = nn.Parameter(torch.zeros(10))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 2),
=======
                self.far_rel_bias.expand(self.n_head, 10),
>>>>>>> REPLACE