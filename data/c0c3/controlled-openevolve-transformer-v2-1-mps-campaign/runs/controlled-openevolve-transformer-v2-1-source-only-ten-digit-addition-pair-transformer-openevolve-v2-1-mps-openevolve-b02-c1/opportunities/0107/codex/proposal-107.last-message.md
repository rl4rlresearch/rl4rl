MECHANISM: Extended head-0 far-distance bias sharing

HYPOTHESIS: Tying head 0’s three farthest relative-distance biases will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because the verified design already ties its two farthest biases and head 1 tolerates substantially broader far-distance sharing.

INTENDED_EDIT: Store one fewer head-0 relative-bias parameter and reconstruct its three farthest distances with the shared zero reference.

EVIDENCE: The 984-parameter design achieved 99.75% while tying head 0’s two farthest distances and using still stronger distance sharing in head 1; extending only the adjacent head-0 boundary is an isolated one-parameter compression that preserves all content projections and final-LayerNorm distinctions implicated by recent failures.

<<<<<<< SEARCH
        # Head 0 ties the two farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 2)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 ties the three farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 3)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 2)),
=======
                F.pad(self.rel_bias[0], (0, 3)),
>>>>>>> REPLACE