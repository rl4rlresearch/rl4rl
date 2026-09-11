MECHANISM: Learned adjacent-boundary bias quintuplet sharing

HYPOTHESIS: Sharing head 1’s fourth- through eighth-farthest relative biases will reduce the model from 1,040 to 1,039 parameters while retaining at least 99% accuracy, because the corresponding quadruplet-sharing model achieved 99.68%.

INTENDED_EDIT: Shorten head 1’s relative-bias parameter by one coordinate and extend its final learned boundary value across four following distances, leaving head 0 and all other behavior unchanged.

EVIDENCE: Successive learned sharing of head 1’s boundary triplet and quadruplet achieved 99.81% and 99.68%, while forcing the fourth-farthest bias to the fixed reference collapsed accuracy to 14.36%; this supports preserving an adaptive shared value while removing one adjacent degree of freedom.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary quadruplet.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 6)),
            ]
        )
=======
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary quintuplet.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 7)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                        self.rel_bias[1][-1:].expand(3),
=======
                        self.rel_bias[1][-1:].expand(4),
>>>>>>> REPLACE