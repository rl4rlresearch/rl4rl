MECHANISM: Shared value/output-projection bias gauge

HYPOTHESIS: Tying the seventh value-bias coordinate to the seventh projection-bias parameter will reduce the model to 1611 parameters while retaining at least 99% accuracy, because it preserves an adaptive position-independent value offset without adding an independent degree of freedom.

INTENDED_EDIT: Store six independent value-bias coordinates and reconstruct the seventh from `proj_bias[-1]`, while retaining the existing fixed-zero eighth value bias.

EVIDENCE: Fixing the seventh value bias to zero reduced accuracy to 81.59%, while sharing the otherwise necessary `ln1` contrast with an existing query bias achieved 100% at 1612 parameters; this tests the same successful adaptive-sharing strategy on the most recent failed reduction.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 1))
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_bias = torch.cat(
            (self.qv_bias[d_model:], self.qv_bias.new_zeros(1))
        )
=======
        v_bias = torch.cat(
            (
                self.qv_bias[d_model:],
                self.proj_bias[-1:],
                self.qv_bias.new_zeros(1),
            )
        )
>>>>>>> REPLACE