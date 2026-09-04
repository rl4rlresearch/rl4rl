MECHANISM: Within-head second value-row normalized-input gauge

HYPOTHESIS: Adding value row 17 to the verified gauges will reduce the model from 1,581 to 1,580 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Omit the final normalized-input coefficient from the second value coordinate of the first attention head, alongside the verified query, key, and value gauges.

EVIDENCE: Gauging value row 16 achieved 99.94% accuracy at 1,581 parameters. Additional gauges in the second head failed sharply, while an adjacent first-head key gauge reached 98.19%, motivating the smallest untested within-first-head value extension.

<<<<<<< SEARCH
        # Retain the verified balanced query and first-key gauges, then gauge
        # the first value coordinate, whose basis can adapt with proj.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
        )
=======
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
        )
>>>>>>> REPLACE