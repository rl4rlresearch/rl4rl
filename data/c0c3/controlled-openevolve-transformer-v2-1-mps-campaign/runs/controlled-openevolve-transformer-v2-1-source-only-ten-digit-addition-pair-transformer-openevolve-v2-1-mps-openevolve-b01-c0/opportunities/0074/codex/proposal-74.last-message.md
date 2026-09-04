MECHANISM: Coordinate-selective value-bias removal

HYPOTHESIS: Fixing only the second value-bias coordinate to zero will reduce the model to 1602 parameters while maintaining at least 99% accuracy, because the failed third-coordinate removal may reflect coordinate-specific importance rather than a need for all three remaining learned value offsets.

INTENDED_EDIT: Store the first and third value-bias coordinates, reconstruct the second as zero, and preserve the existing zero fourth-through-sixth and eighth coordinates plus the seventh-coordinate projection-bias tie.

EVIDENCE: Removing value-bias coordinates four through six successively retained 99.66% or better accuracy, while removing the third reduced accuracy to 96.20%; retaining that sensitive third coordinate while testing the second is the most direct way to distinguish localized coordinate importance from a general capacity limit.

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
                self.qv_bias[d_model : d_model + 1],
                self.qv_bias.new_zeros(1),
                self.qv_bias[d_model + 1 :],
                self.qv_bias.new_zeros(3),
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
>>>>>>> REPLACE