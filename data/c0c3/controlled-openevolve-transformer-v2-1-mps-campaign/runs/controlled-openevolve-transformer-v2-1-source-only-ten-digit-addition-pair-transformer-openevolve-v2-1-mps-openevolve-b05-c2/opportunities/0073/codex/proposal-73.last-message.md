MECHANISM: Asymmetric first-head query-bias anchoring

HYPOTHESIS: Fixing `q_bias[2]` in the qualified 1510-parameter design will yield 1509 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reconstruct query-bias coordinates 0, 1, 2, and 4 as zero while retaining the sole learned `ln1` scale and all remaining query biases.

EVIDENCE: Fixing first-head `q_bias[1]` achieved 99.89% at 1510 parameters, whereas fixing its second-head counterpart `q_bias[5]` fell to 94.06%; this motivates testing another first-head coordinate rather than extending the harmful second-head constraint.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(2),
                self.q_bias[:self.head_dim - 2],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 2:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 3:],
            )
        )
>>>>>>> REPLACE