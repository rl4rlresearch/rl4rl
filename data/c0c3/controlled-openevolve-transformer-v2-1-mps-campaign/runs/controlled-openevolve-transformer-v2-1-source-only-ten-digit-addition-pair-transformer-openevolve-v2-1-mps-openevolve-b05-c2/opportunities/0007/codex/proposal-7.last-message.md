MECHANISM: Pre-MLP LayerNorm offset absorption

HYPOTHESIS: Removing the 8-parameter `ln2` bias will reduce the model from 1591 to 1583 parameters while retaining at least 99% accuracy, because its contribution is exactly absorbable by the existing `fc1` bias without reducing MLP width.

INTENDED_EDIT: Disable only the second LayerNorm’s learned bias while preserving all attention value-bias coordinates, weight matrices, training settings, and decoding behavior.

EVIDENCE: The 1591-parameter design achieved 99.96%, while removing a second value-bias coordinate collapsed accuracy to 70.07%; this motivates preserving the sensitive value-bias pathway and instead removing an algebraically redundant pre-MLP offset.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE