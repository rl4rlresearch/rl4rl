MECHANISM: First second-head key-row normalized-input gauge

HYPOTHESIS: Extending the verified 1,547-parameter design to second-head key row 12 will produce a 1,546-parameter model with at least 99% accuracy.

INTENDED_EDIT: Retain every verified gauge, complete the four first-head key-row gauges, and gauge the adjacent first key row of the second attention head.

EVIDENCE: Successive first-head key extensions achieved 99.91% at 1,549 parameters, 99.96% at 1,548, and 99.90% at 1,547; continuing to adjacent row 12 is the smallest reduction and directly tests the previously fragile head boundary.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, extend the first-head key
        # gauge to its adjacent row, and gauge every first-head value row.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
=======
        # Retain the verified balanced query gauges, gauge every first-head key
        # row plus the adjacent first second-head row, and gauge every
        # first-head value row.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
>>>>>>> REPLACE