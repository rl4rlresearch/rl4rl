MECHANISM: Extended head-0 far-distance bias sharing

HYPOTHESIS: Tying head 0’s third-farthest relative-attention distance to its existing zero reference will reduce the model from 984 to 983 parameters while retaining at least 99% accuracy, because its two farthest distances are already tied successfully and all sensitive LayerNorm, MLP, and key-projection distinctions remain unchanged.

INTENDED_EDIT: Store one fewer head-0 relative-bias parameter and reconstruct its three farthest distances as the shared zero reference.

EVIDENCE: The verified 984-parameter design achieves 99.75% while already tying head 0’s two farthest distances; extending that same isolated positional-bias constraint by one adjacent distance is the smallest untested compression after additional LayerNorm and companion-key sharing failed.

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