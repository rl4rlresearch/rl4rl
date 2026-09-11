MECHANISM: Third second-head key-row normalized-input gauge

HYPOTHESIS: Extending the verified 1,545-parameter design to second-head key row 14 will produce a 1,544-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce all verified first-head key and value gauges plus second-head key rows 12–13, then gauge the adjacent second-head key row 14.

EVIDENCE: The verified progression through key rows 9–13 maintained 99.90%–99.99% accuracy, and row 13 achieved 99.95% at 1,545 parameters; extending the same adjacent-row pattern is the smallest informative reduction.

<<<<<<< SEARCH
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
=======
        # Retain the verified balanced query gauges, gauge every first-head key
        # row plus three adjacent second-head key rows, and gauge every
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
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
>>>>>>> REPLACE