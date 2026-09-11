MECHANISM: Uncoupled fourth value-bias removal

HYPOTHESIS: Fixing the fourth value-bias coordinate to zero will reduce the model to 1603 parameters while maintaining at least 99% accuracy, because value bias contributes a constant attention output that the learned centered projection bias can absorb.

INTENDED_EDIT: Store three independent value-bias coordinates and insert fixed zeros for the fourth, fifth, and sixth coordinates, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.

EVIDENCE: Successive uncoupled removal of the sixth and then fifth value-bias coordinates achieved 99.97% at 1605 parameters and 99.99% at 1604 parameters; removing the adjacent fourth coordinate is the most direct test of whether this demonstrated redundancy extends one coordinate further.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 4))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(2),
                self.proj_bias[-1:],
=======
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(3),
                self.proj_bias[-1:],
>>>>>>> REPLACE