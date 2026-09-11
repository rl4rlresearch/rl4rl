MECHANISM: Sixth noncontiguous LayerNorm-nullspace weight gauge

HYPOTHESIS: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, and 6 will reduce the model from 1,605 to 1,604 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend the successful noncontiguous gauge set with row 6, omitting and reconstructing its final input coefficient through the existing learned linear reparameterization.

EVIDENCE: The current five-row set achieved 99.83% accuracy at 1,605 parameters, while the principal failed extension involved row 3; adding adjacent row 6 is the smallest informative reduction that preserves the successful exclusion of row 3.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5)
=======
        self.gauged_rows = (0, 1, 2, 4, 5, 6)
>>>>>>> REPLACE