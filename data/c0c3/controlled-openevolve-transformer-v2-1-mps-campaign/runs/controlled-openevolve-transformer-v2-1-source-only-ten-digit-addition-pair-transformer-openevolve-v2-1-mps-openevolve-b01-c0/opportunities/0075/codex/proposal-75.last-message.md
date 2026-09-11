MECHANISM: Coordinate-selective first value-bias removal

HYPOTHESIS: Fixing only the first value-bias coordinate to zero will reduce the model to 1602 parameters while maintaining at least 99% accuracy, because the failures from removing the second or third coordinates indicate coordinate-specific importance while leaving the first coordinate as the remaining untested value-bias redundancy.

INTENDED_EDIT: Store the second and third value-bias coordinates, reconstruct the first as zero, and preserve the existing zero fourth-through-sixth and eighth coordinates plus the seventh-coordinate projection-bias tie.

EVIDENCE: The 1603-parameter design reached 99.66% after removing value-bias coordinates four through six, while selective removal of the second scored 14.65% and removal of the third scored 96.20%; testing the first coordinate completes the coordinate-selective comparison without constraining any bias already shown to be important.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 5))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_bias = torch.cat(
            (
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(3),
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
=======
        v_bias = torch.cat(
            (
                self.qv_bias.new_zeros(1),
                self.qv_bias[d_model:],
                self.qv_bias.new_zeros(3),
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
>>>>>>> REPLACE