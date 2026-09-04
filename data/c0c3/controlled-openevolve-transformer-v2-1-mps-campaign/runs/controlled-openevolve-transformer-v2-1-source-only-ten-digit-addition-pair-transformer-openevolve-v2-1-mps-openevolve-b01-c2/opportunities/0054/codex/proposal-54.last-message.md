MECHANISM: Within-head third value-row normalized-input gauge

HYPOTHESIS: Extending the verified 1,580-parameter design with value row 18 will produce a 1,579-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified query rows 0, 1, 4, and 5, key row 8, and value rows 16 and 17, then omit the final normalized-input coefficient from adjacent value row 18.

EVIDENCE: Value rows 16 and 17 successively achieved 99.94% at 1,581 parameters and 99.83% at 1,580 parameters; extending the same successful first-head value-row sequence is the smallest informative reduction.

<<<<<<< SEARCH
        # Balance four query-row gauges across the two attention heads, then
        # gauge the first key row without further constraining either query.
        self.gauged_rows = (0, 1, head_dim, head_dim + 1, d_model)
=======
        # Retain the verified balanced query and first-key gauges, then gauge
        # three adjacent value coordinates in the first attention head.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
        )
>>>>>>> REPLACE