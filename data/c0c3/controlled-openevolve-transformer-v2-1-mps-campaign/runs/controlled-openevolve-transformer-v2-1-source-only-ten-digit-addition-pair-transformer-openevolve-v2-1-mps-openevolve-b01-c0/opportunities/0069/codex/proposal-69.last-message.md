MECHANISM: Uncoupled sixth value-bias removal

HYPOTHESIS: Fixing the sixth value-bias coordinate to zero will reduce the model to 1605 parameters while maintaining at least 99% accuracy, because a value bias contributes only a constant attention output that the learned projection bias can absorb, without coupling it to a query coordinate.

INTENDED_EDIT: Store five independent value-bias coordinates instead of six and insert a fixed zero for the sixth coordinate while preserving the existing seventh-coordinate projection-bias tie and fixed-zero eighth coordinate.

EVIDENCE: The 1606-parameter design achieves 99.89% with the eighth value bias fixed to zero and the seventh shared with projection bias, while sharing the sixth value bias with its query coordinate scored 94.91%; fixing the sixth coordinate isolates whether that failure arose from harmful query/value coupling rather than loss of an independently learned value offset.

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
                self.qv_bias.new_zeros(1),
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
>>>>>>> REPLACE