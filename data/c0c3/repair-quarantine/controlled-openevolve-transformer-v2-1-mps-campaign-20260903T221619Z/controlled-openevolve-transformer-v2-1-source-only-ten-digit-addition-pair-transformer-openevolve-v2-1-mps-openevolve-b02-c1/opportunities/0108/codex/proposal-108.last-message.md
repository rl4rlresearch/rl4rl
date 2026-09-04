MECHANISM: Four-way head-0 far-distance bias sharing

HYPOTHESIS: Tying head 0’s fourth-farthest relative-attention distance to its existing zero reference will reduce the model from 983 to 982 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store one fewer head-0 relative-bias parameter and reconstruct its four farthest distances as the shared zero reference.

EVIDENCE: Extending head 0’s zero-reference sharing from two to three farthest distances achieved 99.94% accuracy at 983 parameters, making one further adjacent extension the smallest evidence-backed compression step.

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
        # Head 0 ties the four farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
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
                F.pad(self.rel_bias[0], (0, 4)),
>>>>>>> REPLACE