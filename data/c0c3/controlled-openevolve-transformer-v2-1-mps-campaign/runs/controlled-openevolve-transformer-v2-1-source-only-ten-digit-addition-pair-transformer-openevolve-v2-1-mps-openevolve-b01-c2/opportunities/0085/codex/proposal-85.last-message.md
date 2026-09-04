MECHANISM: Reduced-chart second-head value-row quotient

HYPOTHESIS: Gauging value row 21 while training it with ordinary reduced-coordinate AdamW will produce a 1,541-parameter model with at least 99% accuracy.

INTENDED_EDIT: Add second-head value row 21 to the normalized-input gauge, while retaining dense-coordinate updates only for key row 15 and value row 20.

EVIDENCE: Dense-coordinate training of row 21 reached 98.28%, while first-head value rows 16–19 succeeded with ordinary reduced-coordinate updates and the current 1,542-parameter design reached 100%; this isolates optimizer geometry as the next variable.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every key row,
        # gauge every first-head value row, and extend the same quotient to
        # the adjacent first value row of the second head.
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
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
        )
=======
        # Retain the verified balanced query gauges, gauge every key row and
        # every first-head value row, then gauge the first two value rows of
        # the second head. Row 21 uses the ordinary reduced-coordinate update.
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
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
        )
>>>>>>> REPLACE