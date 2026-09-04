MECHANISM: Adjacent head-0 boundary-bias sharing

HYPOTHESIS: Tying head 0’s load-bearing fourth-farthest relative bias to its nearest learned neighbor will reduce the model from 983 to 982 parameters while retaining at least 99% accuracy, because it preserves a learned nonzero boundary contrast rather than collapsing that distance to the zero reference.

INTENDED_EDIT: Store one fewer head-0 relative-bias parameter and reuse its final learned bias for both the fourth- and fifth-farthest distances.

EVIDENCE: Three zero-referenced farthest distances achieved 99.94% at 983 parameters, whereas extending the zero-reference group to the fourth-farthest distance collapsed accuracy to 14.19%; adjacent sharing directly tests whether that distance needs a nonzero learned contrast rather than an independent scalar.

<<<<<<< SEARCH
        # Head 0 ties the three farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 3)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 ties the three farthest distances to its reference and
        # shares the next boundary pair. Head 1 ties the three farthest to
        # its reference, shares the next boundary quadruplet, and
        # reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 4)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 3)),
=======
                torch.cat(
                    [
                        self.rel_bias[0],
                        self.rel_bias[0][-1:],
                        self.rel_bias[0].new_zeros(3),
                    ]
                ),
>>>>>>> REPLACE