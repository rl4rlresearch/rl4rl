MECHANISM: Uncoupled fifth value-bias removal

HYPOTHESIS: Fixing the fifth value-bias coordinate to zero will reduce the model to 1604 parameters while maintaining at least 99% accuracy, because value bias contributes only a constant attention output that the learned centered projection bias can absorb.

INTENDED_EDIT: Store four independent value-bias coordinates and insert fixed zeros for the fifth and sixth coordinates, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.

EVIDENCE: Fixing the sixth value-bias coordinate independently to zero achieved 99.97% at 1605 parameters, whereas coupling it to the aligned query bias achieved only 94.91%; extending the successful uncoupled removal to the adjacent fifth value coordinate is the most direct test of the constant-output redundancy.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 3))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(1),
                self.proj_bias[-1:],
=======
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(2),
                self.proj_bias[-1:],
>>>>>>> REPLACE