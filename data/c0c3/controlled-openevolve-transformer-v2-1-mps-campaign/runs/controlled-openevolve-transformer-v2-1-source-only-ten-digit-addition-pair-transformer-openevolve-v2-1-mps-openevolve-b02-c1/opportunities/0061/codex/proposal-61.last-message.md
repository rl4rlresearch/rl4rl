MECHANISM: Learned adjacent-boundary bias triplet sharing

HYPOTHESIS: Sharing head 1’s fourth-, fifth-, and sixth-farthest learned relative biases will reduce the model from 1,042 to 1,041 parameters while retaining at least 99% accuracy, because sharing the fourth and fifth biases already achieved 99.38%, whereas forcing the fourth-farthest bias to the fixed reference failed.

INTENDED_EDIT: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct its two following boundary biases from the final learned value; leave head 0 and all other behavior unchanged.

EVIDENCE: The 1,042-parameter adjacent-sharing design achieved 99.38% accuracy after the direct fourth-distance reference tie collapsed to 14.36%, indicating that this boundary region needs an adaptive value but may not need independent neighboring coordinates.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary pair.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 4)),
            ]
        )
=======
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary triplet.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 5)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                    [
                        self.rel_bias[1],
                        self.rel_bias[1][-1:],
                        self.rel_bias[1].new_zeros(3),
                    ]
=======
                    [
                        self.rel_bias[1],
                        self.rel_bias[1][-1:].expand(2),
                        self.rel_bias[1].new_zeros(3),
                    ]
>>>>>>> REPLACE