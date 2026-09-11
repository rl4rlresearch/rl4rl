MECHANISM: Fourth first-head key-row normalized-input gauge

HYPOTHESIS: Extending the verified 1,548-parameter design to first-head key row 11 will produce a 1,547-parameter model with at least 99% accuracy.

INTENDED_EDIT: Gauge the remaining first-head key row while retaining all previously verified query, value, output, embedding, and positional quotients.

EVIDENCE: Successive first-head key-row extensions reached 99.91% at 1,549 parameters and 99.96% at 1,548 parameters; row 11 completes the same adjacent first-head pattern without imposing a fragile second-head constraint.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, extend the first-head key
        # gauge to three adjacent rows, and gauge every first-head value row.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            2 * d_model,
=======
        # Retain the verified balanced query gauges, gauge every first-head key
        # row, and gauge every first-head value row.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            2 * d_model,
>>>>>>> REPLACE