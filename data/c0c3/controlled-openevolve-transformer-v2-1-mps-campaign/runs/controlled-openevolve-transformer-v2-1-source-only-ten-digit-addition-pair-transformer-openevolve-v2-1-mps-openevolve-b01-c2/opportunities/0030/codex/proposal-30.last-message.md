MECHANISM: Affine-free MLP normalization with exact all-row nullspace gauge

HYPOTHESIS: Removing the redundant learned `ln2` scale and gauge-fixing every `fc1` row will produce a 1,591-parameter model with at least 99% accuracy.

INTENDED_EDIT: Make the MLP’s pre-normalization parameter-free and extend `fc1` gauge fixing to all 12 rows, preserving the initial function while making the common-coefficient null direction exact throughout training.

EVIDENCE: The 1,600-parameter model achieved 99.91%, but separately gauging row 11 fell to 98.40% and earlier gauging row 3 fell to 97.77%; these failures suggest interaction with the learned `ln2` scale, which prevents the current row-wise gauge from remaining an exact null direction after initialization.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10)
=======
        self.gauged_rows = tuple(range(out_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
=======
        # fc1 can absorb LayerNorm's feature scales. Removing the affine scale
        # also makes every fc1 common-row coefficient an exact null direction.
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE