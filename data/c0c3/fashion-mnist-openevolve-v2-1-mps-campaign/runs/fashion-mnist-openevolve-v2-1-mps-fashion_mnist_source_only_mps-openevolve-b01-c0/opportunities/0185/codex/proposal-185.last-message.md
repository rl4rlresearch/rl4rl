MECHANISM: Argmax-invariant temperature plateau escape

HYPOTHESIS: Lowering the ensemble temperature to 0.717 will preserve all 9,290 predictions while producing a measurable cross-entropy reduction if the ensemble remains underconfident.

INTENDED_EDIT: Apply a larger, still conservative temperature decrement after the one-ULP change was hidden by metric quantization.

EVIDENCE: The adjacent-lower-float32 temperature preserved 9,290 correct predictions but left reported cross-entropy unchanged; positive temperature scaling preserves argmax, so a larger decrement is the safest informative tie-breaker probe.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717143714427948
=======
        return ensemble_log_probabilities / 0.717
>>>>>>> REPLACE