MECHANISM: Second-row QKV LayerNorm-nullspace gauge

HYPOTHESIS: Gauge-fixing QKV row 1 alongside the verified row 0 gauge will reduce the model to 1,585 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Omit and reconstruct the final input coefficient of QKV rows 0 and 1, preserving their initialized functions while leaving the remaining QKV rows unconstrained.

EVIDENCE: The current single-row QKV gauge achieved 99.94% accuracy at 1,586 parameters, substantially above threshold; adding one adjacent query-row gauge is the smallest incremental reduction along this successful path.

<<<<<<< SEARCH
        self.gauged_rows = (0,)
        self.ungauged_rows = tuple(range(1, self.out_features))
=======
        self.gauged_rows = (0, 1)
        self.ungauged_rows = tuple(range(2, self.out_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias and one representative of QKV row 0's normalized-
        # input coefficient gauge.
=======
        # Retain query bias and representatives of QKV rows 0 and 1's
        # normalized-input coefficient gauges.
>>>>>>> REPLACE

<<<<<<< SEARCH
                # All attention scales initialize to one, so subtracting the
                # omitted coefficient preserves the initial row function.
=======
                # All attention scales initialize to one, so subtracting each
                # omitted coefficient preserves the initial row functions.
>>>>>>> REPLACE