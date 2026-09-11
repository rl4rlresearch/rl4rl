MECHANISM: Reduced-chart first-head query-row quotient

HYPOTHESIS: Gauging first-head query row 2 with ordinary reduced-coordinate AdamW will reduce the model to 1,538 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add query row 2 to the normalized-input gauge; existing custom QKV optimization leaves it on the reduced-coordinate path while preserving dense updates for sensitive rows 15, 20, and 23.

EVIDENCE: The 1,539-parameter design reached 99.95%, and ordinary reduced-coordinate updates successfully trained adjacent normalized-input gauges including all first-head value rows and second-head value rows 21–22; extending the already verified query-row 0–1 pattern is the smallest informative reduction.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges and gauge every key and
        # value row. Final rows 15 and 23 and value row 20 use recovered
        # dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            head_dim,
=======
        # Extend the verified first-head query gauges through row 2 and retain
        # every key and value gauge. Final rows 15 and 23 and value row 20 use
        # recovered dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            head_dim,
>>>>>>> REPLACE