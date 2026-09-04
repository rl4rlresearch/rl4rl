MECHANISM: Lower-adjacent float32 temperature bracketing

HYPOTHESIS: The next representable float32 temperature below the current value will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924331665039.

INTENDED_EDIT: Replace the ensemble temperature with the exact lower-adjacent float32 value.

EVIDENCE: The upper-adjacent float32 temperature increased cross-entropy to 0.20249243392944336, while the lower-adjacent trial only timed out and produced no contrary metric evidence; it remains the unresolved side of the local optimum.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.71714375
=======
        return ensemble_log_probabilities / 0.717143714427948
>>>>>>> REPLACE