MECHANISM: Second value/output-bias redundancy

HYPOTHESIS: Fixing a second value-bias coordinate to zero will reduce the model to 1611 parameters while retaining at least 99% accuracy, because attention’s position-independent value bias is representable by the learned output-projection bias.

INTENDED_EDIT: Store six value-bias coordinates instead of seven and reconstruct the final two coordinates as fixed zeros, leaving all query biases and the shared `ln1` contrast unchanged.

EVIDENCE: The 1612-parameter design achieved 100% accuracy while already fixing the final value-bias coordinate and retaining a seven-degree centered projection bias; extending that successful redundancy by one adjacent coordinate is the smallest informative reduction.

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
            (self.qv_bias[d_model:], self.qv_bias.new_zeros(2))
        )
>>>>>>> REPLACE