MECHANISM: Paired per-head query-bias gauge fixing

HYPOTHESIS: Fixing `q_bias[0]` alongside the qualified fixed `q_bias[4]` will reduce the model to 1511 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one learned query-bias scalar and reconstruct the first coordinate of each attention head as zero.

EVIDENCE: Fixing `q_bias[4]` qualified at 99.88% and remained successful in the 1512-parameter model at 99.96%; the failed 1511 attempt instead removed the final learned `ln1` scale and fell to 74.13%, motivating a distinct query-key gauge.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                self.q_bias[:self.head_dim],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(1),
                self.q_bias[:self.head_dim - 1],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 1:],
            )
        )
>>>>>>> REPLACE