MECHANISM: Extended head-0 far-distance bias sharing with reduced validation overhead

HYPOTHESIS: Tying head 0’s six farthest relative-distance biases will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy; evaluating every 500 steps will allow the previously timed-out configuration to finish all 4,999 training steps.

INTENDED_EDIT: Remove one head-0 relative-bias parameter, reconstruct six farthest distances with the zero reference, and reduce intermediate validation frequency without changing optimization steps.

EVIDENCE: Successive extensions through five tied distances retained at least 99.89% accuracy, while the six-distance attempt timed out rather than producing negative accuracy evidence; its negligible model-side compute change makes reducing validation overhead the most direct retry.

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

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=500)
>>>>>>> REPLACE