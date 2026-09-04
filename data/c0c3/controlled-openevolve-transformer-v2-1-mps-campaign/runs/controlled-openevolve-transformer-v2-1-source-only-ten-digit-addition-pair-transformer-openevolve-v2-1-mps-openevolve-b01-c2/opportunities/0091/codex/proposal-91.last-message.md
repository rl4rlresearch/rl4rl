MECHANISM: Reduced-chart second-head query-row quotient

HYPOTHESIS: Extending the verified 1,537-parameter design by gauging second-head query row 6 with ordinary reduced-coordinate AdamW will produce a 1,536-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce every verified key, value, and first-head query gauge, add second-head query row 6, and retain dense-coordinate updates for sensitive key row 15 and value rows 20 and 23.

EVIDENCE: First-head query rows 2 and 3 successively achieved 99.93% and 99.95% with reduced-coordinate AdamW; row 6 is the adjacent untested query row and shares the already-fixed local query-bias coordinate.

<<<<<<< SEARCH
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
=======
        # Complete the first-head query gauges and extend the second head
        # through local query row 2. Retain every verified key and value gauge;
        # sensitive rows 15, 20, and 23 use dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 2,
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
            2 * d_model + head_dim + 2,
            2 * d_model + head_dim + 3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. The
    # completed final key row and new second-head value row additionally use
    # dense moments for their omitted normalized-input coefficients.
=======
    # Residual-output quotients retain dense output-coordinate moments. Key
    # row 15 and value rows 20 and 23 retain their verified dense updates;
    # query rows 2, 3, and 6 use ordinary reduced-coordinate AdamW.
>>>>>>> REPLACE

<<<<<<< SEARCH
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                ),
=======
                (
                    2 * qkv.in_features - 1,
                    2 * qkv.in_features + block.attn.head_dim,
                    3 * qkv.in_features - 1,
                ),
>>>>>>> REPLACE