MECHANISM: Cross-projection normalized-input gauge fixing

HYPOTHESIS: Adding a gauge for the first key row to the verified four balanced query-row gauges will reduce the model from 1,583 to 1,582 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Retain query-row gauges 0, 1, 4, and 5, and omit the final normalized-input coefficient from fused QKV row 8, the first key coordinate of the first attention head.

EVIDENCE: The four balanced query gauges achieved 99.98% accuracy at 1,583 parameters, while adding query row 6 fell to 65.68%; testing one key-row gauge is the smallest orthogonal reduction and avoids further constraining either head’s query representation.

<<<<<<< SEARCH
        # Balance four query-row gauges across the two attention heads.
        self.gauged_rows = (0, 1, head_dim, head_dim + 1)
=======
        # Balance four query-row gauges across the two attention heads, then
        # gauge the first key row without further constraining either query.
        self.gauged_rows = (0, 1, head_dim, head_dim + 1, d_model)
>>>>>>> REPLACE