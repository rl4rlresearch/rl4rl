MECHANISM: Asymmetric fourth-distance boundary tie

HYPOTHESIS: Tying the fourth-farthest relative-distance bias only in head 1 will reduce the model from 1,043 to 1,042 parameters while retaining at least 99% accuracy, because head 1 already tolerated the third-distance tie while head 0 retained the positional coordinate whose removal from both heads caused failure.

INTENDED_EDIT: Extend head 1’s boundary tie from its three farthest relative distances to its four farthest distances, leaving head 0’s successful two-distance tie and all other model, optimization, checkpointing, and decoding behavior unchanged.

EVIDENCE: Tying three farthest distances in both heads collapsed accuracy to 19.01%, but tying the third-farthest distance only in head 1 achieved 99.73% at 1,043 parameters; this isolates another one-parameter ablation in the head that has demonstrated greater boundary redundancy.

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
        # Head 0 ties the two farthest distances, while head 1 ties the
        # four farthest distances to its softmax-invisible reference.
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
                F.pad(self.rel_bias[1], (0, 4)),
            ]
        )
>>>>>>> REPLACE