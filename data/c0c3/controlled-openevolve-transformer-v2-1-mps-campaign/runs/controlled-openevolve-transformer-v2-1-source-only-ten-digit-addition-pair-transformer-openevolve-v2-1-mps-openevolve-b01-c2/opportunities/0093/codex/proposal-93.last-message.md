MECHANISM: Reduced-chart final second-head query-row quotient

HYPOTHESIS: Gauging second-head query row 7 with ordinary reduced-coordinate AdamW will produce a 1,536-parameter model with at least 99% accuracy.

INTENDED_EDIT: Add query row 7 to the normalized-input gauge while retaining dense-coordinate updates only for sensitive key row 15 and value rows 20 and 23.

EVIDENCE: First-head query rows 2–3 succeeded with reduced-coordinate AdamW at 1,537 parameters; unlike failed row 6, row 7 retains its learned query bias, isolating whether row 6’s fixed-bias coordinate caused its narrow 98.89% miss.

<<<<<<< SEARCH
        # Complete the first-head query-row gauges and retain every verified
        # key and value gauge. Sensitive rows 15, 20, and 23 use recovered
        # dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            d_model,
=======
        # Gauge the final second-head query row while retaining every verified
        # key and value gauge. Sensitive rows 15, 20, and 23 use recovered
        # dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 3,
            d_model,
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve ordinary reduced-coordinate AdamW for all other QKV
        # coefficients, including query rows 2–3 and value rows 21–22.
=======
        # Preserve ordinary reduced-coordinate AdamW for all other QKV
        # coefficients, including query rows 2–3 and 7 and value rows 21–22.
>>>>>>> REPLACE

<<<<<<< SEARCH
    # row 15 and value rows 20 and 23 retain their verified dense-coordinate
    # updates; query rows 2–3 and value rows 21–22 use the reduced chart.
=======
    # row 15 and value rows 20 and 23 retain their verified dense-coordinate
    # updates; query rows 2–3 and 7 and value rows 21–22 use the reduced chart.
>>>>>>> REPLACE