MECHANISM: Uncoupled third value-bias removal

HYPOTHESIS: Fixing the third value-bias coordinate to zero will reduce the model to 1602 parameters while maintaining at least 99% accuracy, because value bias contributes a constant attention output that the learned centered projection bias can absorb.

INTENDED_EDIT: Store two independent value-bias coordinates and insert fixed zeros for the third through sixth coordinates, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.

EVIDENCE: Successive uncoupled removal of the sixth, fifth, and fourth value-bias coordinates achieved 99.97%, 99.99%, and 99.66% accuracy respectively; extending the same demonstrated redundancy to the adjacent third coordinate is the most informative one-parameter reduction.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 5))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(3),
                self.proj_bias[-1:],
=======
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(4),
                self.proj_bias[-1:],
>>>>>>> REPLACE