MECHANISM: Adjacent value/output-projection bias sharing

HYPOTHESIS: Tying the sixth value-bias coordinate to the sixth projection-bias parameter will reduce the model to 1610 parameters while retaining at least 99% accuracy, because the successful seventh-coordinate tie shows that adaptive sharing can preserve capacity lost when a value bias is fixed.

INTENDED_EDIT: Store five independent value-bias coordinates and reconstruct the sixth and seventh coordinates from the final two projection-bias parameters, retaining the fixed-zero eighth coordinate.

EVIDENCE: Tying the seventh value bias to `proj_bias[-1]` achieved 99.98% at 1611 parameters, whereas fixing that value bias to zero achieved only 81.59%; extending the same sharing mechanism to the adjacent sixth coordinate is the smallest evidence-backed reduction.

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
                self.proj_bias[-2:],
                self.qv_bias.new_zeros(1),
            )
        )
>>>>>>> REPLACE