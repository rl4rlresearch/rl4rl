MECHANISM: Reduced-chart final first-head query-row quotient

HYPOTHESIS: Extending the verified 1,538-parameter design by gauging first-head query row 3 with ordinary reduced-coordinate AdamW will produce a 1,537-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified query-row-2 and final-value-row gauges, then gauge adjacent query row 3 while retaining dense-coordinate updates for sensitive key row 15 and value rows 20 and 23.

EVIDENCE: Query row 2 achieved 99.93% at 1,538 parameters with reduced-coordinate AdamW, while all first-head key and value rows were also successfully gauged; this supports completing the adjacent first-head query row in the same reduced chart.

<<<<<<< SEARCH
        # Retain the verified balanced query gauges, gauge every key row and
        # every first-head value row, then gauge the first three value rows of
        # the second head. Rows 21–22 use reduced-coordinate AdamW.
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
            2 * d_model + head_dim + 2,
        )
=======
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
        # Preserve ordinary reduced-coordinate AdamW for all other QKV
        # coefficients, including newly gauged value rows 21 and 22.
=======
        # Preserve ordinary reduced-coordinate AdamW for all other QKV
        # coefficients, including query rows 2–3 and value rows 21–22.
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Update key row 15 and value row 20 in eight-coordinate geometry,
        # then project them back into the zero-final-coefficient chart.
=======
        # Update key row 15 and value rows 20 and 23 in eight-coordinate
        # geometry, then project them back into the anchored chart.
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Residual-output quotients retain dense output-coordinate moments. Key
    # row 15 and value row 20 retain their verified dense-coordinate updates;
    # newly gauged value rows 21 and 22 use the reduced chart.
=======
    # Residual-output quotients retain dense output-coordinate moments. Key
    # row 15 and value rows 20 and 23 retain their verified dense-coordinate
    # updates; query rows 2–3 and value rows 21–22 use the reduced chart.
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