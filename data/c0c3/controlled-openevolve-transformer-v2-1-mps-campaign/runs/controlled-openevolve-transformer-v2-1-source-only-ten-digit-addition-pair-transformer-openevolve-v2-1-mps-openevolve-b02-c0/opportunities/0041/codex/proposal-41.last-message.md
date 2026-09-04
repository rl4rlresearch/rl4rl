MECHANISM: Independent post-attention LayerNorm scale tying

HYPOTHESIS: Tying one `ln2` scale coordinate to its first learned coordinate will reduce the model from 1,626 to 1,625 parameters while retaining at least 99% accuracy, because the corresponding single dynamic scale tie in `ln1` achieved 99.78% and preserves the all-ones initialization.

INTENDED_EDIT: Reuse `AnchoredLayerNorm` for `ln2`, retaining its successful zero bias anchor while dynamically reconstructing the final scale coordinate from the first.

EVIDENCE: A single learned `ln1` scale tie met the threshold at 99.78%, and the independent `ln2` bias anchor reached 99.95% after an additional within-`ln1` bias tie failed; this motivates applying one proven scale tie to the separate `ln2` normalization rather than further constraining `ln1`.

<<<<<<< SEARCH
        self.ln2 = BiasAnchoredLayerNorm(cfg.d_model)
=======
        self.ln2 = AnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE