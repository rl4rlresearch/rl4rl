MECHANISM: Learned adjacent-boundary bias quadruplet sharing

HYPOTHESIS: Sharing head 1’s fourth- through seventh-farthest relative biases will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy, because the adjacent-boundary triplet achieved 99.81% and this removes only one neighboring positional degree of freedom.

INTENDED_EDIT: Shorten head 1’s relative-bias parameter by one coordinate and extend its final learned boundary value across three following distances, leaving head 0 and all other behavior unchanged.

EVIDENCE: Sharing head 1’s fourth-, fifth-, and sixth-farthest biases achieved 99.81% at 1,041 parameters, substantially above the threshold; extending that successful learned sharing pattern by one adjacent distance is the smallest informative ablation.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary triplet.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 5)),
            ]
        )
=======
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary quadruplet.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 6)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                        self.rel_bias[1][-1:].expand(2),
=======
                        self.rel_bias[1][-1:].expand(3),
>>>>>>> REPLACE