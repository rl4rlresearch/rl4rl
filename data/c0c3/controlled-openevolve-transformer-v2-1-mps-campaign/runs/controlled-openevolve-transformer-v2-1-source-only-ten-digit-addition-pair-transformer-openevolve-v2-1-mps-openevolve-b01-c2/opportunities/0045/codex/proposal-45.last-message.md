MECHANISM: Balanced cross-head query-row gauge fixing

HYPOTHESIS: Gauge-fixing query row 5 alongside verified rows 0, 1, and 4 will produce a 1,583-parameter model with at least 99% accuracy.

INTENDED_EDIT: Add the second query coordinate of the second attention head to the normalized-input QKV gauge, balancing two gauged query rows per head.

EVIDENCE: The cross-head rows 0, 1, and 4 design achieved 99.93% at 1,584 parameters, while concentrating the third gauge on row 2 fell to 95.90%; extending the successful balanced placement to row 5 is the smallest informative reduction.

<<<<<<< SEARCH
        # Keep the two verified gauges and place the third on the first query
        # coordinate of the second head rather than query row 2.
        self.gauged_rows = (0, 1, head_dim)
=======
        # Balance four query-row gauges across the two attention heads.
        self.gauged_rows = (0, 1, head_dim, head_dim + 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias while gauge-fixing normalized-input query rows
        # 0 and 1 in the first head and row 4 in the second head.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
=======
        # Retain query bias while gauge-fixing two normalized-input query rows
        # in each attention head.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE