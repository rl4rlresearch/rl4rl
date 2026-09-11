MECHANISM: Extended head-0 far-distance bias sharing

HYPOTHESIS: Tying head 0’s five farthest relative-distance biases will reduce the model from 982 to 981 parameters while retaining at least 99% accuracy, because successive extensions from two through four tied farthest distances achieved at least 99.97%.

INTENDED_EDIT: Store one fewer head-0 relative-bias parameter and reconstruct its five farthest distances with the shared zero reference.

EVIDENCE: Extending head 0’s tied region from three to four farthest distances achieved 99.97% at 982 parameters, following 99.98% with three tied distances; this continues the same isolated one-parameter compression without altering content projections or LayerNorm distinctions implicated by prior failures.

<<<<<<< SEARCH
        # Head 0 ties the four farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 4)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 ties the five farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 5)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 4)),
=======
                F.pad(self.rel_bias[0], (0, 5)),
>>>>>>> REPLACE