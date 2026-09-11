MECHANISM: Second second-head key-row normalized-input gauge

HYPOTHESIS: Extending the verified 1,546-parameter design to second-head key row 13 will produce a 1,545-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce all verified first-head key and value gauges plus second-head key row 12, then gauge the adjacent second-head key row 13.

EVIDENCE: The verified progression from key rows 9–12 maintained 99.90%–99.99% accuracy, with row 12 achieving 99.99% at 1,546 parameters; extending the same adjacent-row pattern is the smallest informative reduction.

<<<<<<< SEARCH
        # Retain the verified balanced query and first-key gauges, then gauge
        # two adjacent value coordinates in the first head.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
=======
        # Retain the verified balanced query gauges, gauge every first-head key
        # row plus two adjacent second-head key rows, and gauge every
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
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
        )
>>>>>>> REPLACE