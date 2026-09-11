MECHANISM: Complete first-head query-bias anchoring

HYPOTHESIS: Fixing the remaining first-head query-bias coordinate `q_bias[3]` in the qualified 1509-parameter design will yield 1508 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct all four first-head query biases and the first second-head query bias as zero, while retaining the three remaining second-head biases and the independent learned `ln1` scale.

EVIDENCE: Fixing first-head coordinates 1 and 2 successively qualified at 99.89% and 99.92%, whereas fixing second-head coordinate 5 fell to 94.06%; this supports extending the successful asymmetric first-head anchoring direction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(1),
                self.q_bias[:self.head_dim - 1],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 1:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(self.head_dim + 1),
                self.q_bias,
            )
        )
>>>>>>> REPLACE