MECHANISM: Third first-head value-row LayerNorm gauge

HYPOTHESIS: Gauge-fixing value row 18’s final normalized-input coefficient will reduce the model to 1,551 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend the seven verified QKV row gauges to local value coordinate 2 in the first attention head.

EVIDENCE: The 1,552-parameter design reached 99.8% with adjacent first-head value rows 16 and 17 already gauged; extending that successful local pattern avoids the previously reported failures from additional second-head constraints.

<<<<<<< SEARCH
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
            2 * d_model + 1,
        )
=======
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