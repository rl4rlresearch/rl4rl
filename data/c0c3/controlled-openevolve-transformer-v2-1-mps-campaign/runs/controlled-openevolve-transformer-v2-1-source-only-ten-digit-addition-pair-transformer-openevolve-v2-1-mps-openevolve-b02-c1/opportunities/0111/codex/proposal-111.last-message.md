MECHANISM: Extended head-0 far-distance bias sharing

HYPOTHESIS: Tying head 0’s six farthest relative-distance biases will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy, because each successive extension from two through five tied distances remained above 99.89%.

INTENDED_EDIT: Store one fewer head-0 relative-bias parameter and reconstruct its six farthest distances with the shared zero reference.

EVIDENCE: The immediately preceding extension to five tied distances achieved 99.89% at 981 parameters, after extensions to three and four distances achieved 99.98% and 99.97%; continuing this isolated constraint is the smallest evidence-backed compression.

<<<<<<< SEARCH
        # Head 0 ties the five farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 5)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
=======
        # Head 0 ties the six farthest distances. Head 1 ties the three
        # farthest to its reference, shares the next boundary quadruplet,
        # and reconstructs the two preceding transition distances.
        self.rel_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(max_seq_len - 6)),
                nn.Parameter(torch.zeros(max_seq_len - 8)),
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.rel_bias[0], (0, 5)),
=======
                F.pad(self.rel_bias[0], (0, 6)),
>>>>>>> REPLACE