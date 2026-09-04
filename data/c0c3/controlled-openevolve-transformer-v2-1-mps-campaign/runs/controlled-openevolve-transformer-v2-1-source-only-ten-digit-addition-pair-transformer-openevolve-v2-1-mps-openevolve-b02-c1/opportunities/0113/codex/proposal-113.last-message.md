MECHANISM: Six-distance head-0 bias sharing with final-only validation

HYPOTHESIS: Tying head 0’s six farthest relative-distance biases will reduce the model from 981 to 980 parameters while retaining at least 99% accuracy; validating only at initialization and the final step will eliminate enough evaluation overhead for the previously timed-out configuration to complete all 4,999 updates.

INTENDED_EDIT: Remove one head-0 relative-bias parameter, reconstruct its six farthest distances with the zero reference, and set the default evaluation interval equal to the training length so checkpoints are evaluated at step 0 and step 4,999.

EVIDENCE: Successive extensions through five tied distances retained at least 99.89% accuracy. Both six-distance attempts timed out without negative accuracy evidence, including a retry at 500-step validation intervals, so a stronger reduction in validation overhead is the most direct informative test.

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
    p.add_argument("--eval-interval", type=int, default=5000)
>>>>>>> REPLACE