MECHANISM: Coordinate-aligned query/value bias sharing

HYPOTHESIS: Reusing the sixth query-bias coordinate as the sixth value-bias coordinate will reduce the model to 1605 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Store five independent value-bias coordinates and reconstruct the sixth from `qv_bias[5]`, while retaining the seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.

EVIDENCE: Tying the sixth value bias to a projection-bias parameter failed at 28.31%, but coordinate-aligned sharing of the third and fifth normalization contrasts with query biases achieved 99.89% at 1606 parameters; this tests whether the successful aligned query-sharing mechanism preserves the required adaptive sixth value offset.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 2))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_bias = torch.cat(
            (
                self.qv_bias[d_model:],
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
=======
        v_bias = torch.cat(
            (
                self.qv_bias[d_model:],
                self.qv_bias[5:6],
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
>>>>>>> REPLACE