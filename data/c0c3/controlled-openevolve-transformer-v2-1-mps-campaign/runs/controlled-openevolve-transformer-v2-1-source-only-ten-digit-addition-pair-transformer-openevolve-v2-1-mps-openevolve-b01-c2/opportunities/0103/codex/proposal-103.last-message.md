MECHANISM: Complete second-head query-row quotient

HYPOTHESIS: Gauging the remaining second-head query row 6 with reduced-coordinate AdamW in the verified 1,513-parameter position-scalar design will produce a 1,512-parameter model with at least 99% accuracy.

INTENDED_EDIT: Add query row 6 to the normalized-input QKV gauge while retaining dense-coordinate updates for sensitive key row 15 and value rows 20 and 23.

EVIDENCE: Query row 6 previously reached 98.89%, only 0.11 percentage points below the requirement, while the newer dense-position quotient changed the optimization geometry and achieved 99.95% at 1,513 parameters.

<<<<<<< SEARCH
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
=======
        # Gauge every query row while retaining every verified key and value
        # gauge. Sensitive rows 15, 20, and 23 use recovered dense-coordinate
        # AdamW moments; query row 6 uses the ordinary reduced update.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 2,
            head_dim + 3,
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve ordinary reduced-coordinate AdamW for all other QKV
        # coefficients, including query rows 2–3 and 7 and value rows 21–22.
=======
        # Preserve ordinary reduced-coordinate AdamW for all other QKV
        # coefficients, including query rows 2–3 and 6–7 and value rows 21–22.
>>>>>>> REPLACE