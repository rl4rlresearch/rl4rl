MECHANISM: Learned adjacent-boundary bias sharing

HYPOTHESIS: Sharing head 1’s fourth- and fifth-farthest learned relative biases will reduce the model from 1,043 to 1,042 parameters while retaining at least 99% accuracy, because it preserves an adaptive boundary value instead of forcing the load-bearing fourth-farthest bias to the softmax reference.

INTENDED_EDIT: Shorten head 1’s relative-bias parameter by one coordinate and reconstruct its fourth-farthest bias from the neighboring fifth-farthest learned value; leave head 0 and all other behavior unchanged.

EVIDENCE: Head 1 tolerated tying its three farthest biases at 99.73%, but tying the fourth directly to the reference collapsed accuracy to 14.36%; adjacent sharing tests whether that distance needs a learned value without requiring an independent parameter.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances, while head 1 ties the
        # three farthest distances to its softmax-invisible reference.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 3)),
            ]
        )
=======
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference and shares the next boundary pair.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 4)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        rel_bias = torch.stack(
            [
                F.pad(self.rel_bias[0], (0, 2)),
                F.pad(self.rel_bias[1], (0, 3)),
            ]
        )
=======
        rel_bias = torch.stack(
            [
                F.pad(self.rel_bias[0], (0, 2)),
                torch.cat(
                    [
                        self.rel_bias[1],
                        self.rel_bias[1][-1:],
                        self.rel_bias[1].new_zeros(3),
                    ]
                ),
            ]
        )
>>>>>>> REPLACE