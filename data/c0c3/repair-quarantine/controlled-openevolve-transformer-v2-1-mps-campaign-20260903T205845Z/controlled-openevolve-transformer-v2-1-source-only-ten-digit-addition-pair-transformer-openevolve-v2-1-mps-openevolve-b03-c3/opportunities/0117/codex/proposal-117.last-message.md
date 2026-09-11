MECHANISM: Balanced sixth query-bias coordinate pruning

HYPOTHESIS: A 1,372-parameter transformer will retain at least 99% accuracy because the 1,373-parameter design achieved 99.83%, and each preceding single-coordinate query-bias reduction from 1,376 through 1,373 parameters remained above the threshold.

INTENDED_EDIT: Learn only the first query-bias coordinate of each head and reconstruct the remaining six coordinates as zero.

EVIDENCE: The current asymmetric one-plus-two-coordinate query bias reached 99.83% at 1,373 parameters; symmetrically pruning the second head’s corresponding coordinate is the closest informative one-parameter reduction.

<<<<<<< SEARCH
        # Retain one learned bias coordinate in the first head and two in the
        # second, fixing the other five initially zero coordinates.
        self.q_bias = nn.Parameter(
            torch.zeros(d_model - 2 * n_head - 1)
        )
=======
        # Retain one learned bias coordinate in each head, fixing the other
        # six initially zero coordinates.
        self.q_bias = nn.Parameter(
            torch.zeros(d_model - 3 * n_head)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                first_learned,
                self.q_bias.new_zeros(3),
                second_learned,
                self.q_bias.new_zeros(2),
            )
        )
=======
        q_bias = torch.cat(
            (
                first_learned,
                self.q_bias.new_zeros(3),
                second_learned,
                self.q_bias.new_zeros(3),
            )
        )
>>>>>>> REPLACE