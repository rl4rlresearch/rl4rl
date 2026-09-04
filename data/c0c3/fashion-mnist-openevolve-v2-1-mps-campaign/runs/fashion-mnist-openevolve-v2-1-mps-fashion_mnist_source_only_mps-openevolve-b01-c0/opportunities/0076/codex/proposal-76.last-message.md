MECHANISM: Lower-adjacent float32 temperature retry

HYPOTHESIS: The next representable float32 temperature below the current value will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20249243125915528.

INTENDED_EDIT: Replace the final arithmetic-ensemble temperature with its exact lower-adjacent float32 value, 0.7171436548233032.

EVIDENCE: The preceding lower-adjacent move improved cross-entropy while preserving all predictions; the only verification of this next neighbor exceeded the time limit, so it remains without an accepted objective result.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717143714427948
=======
        return ensemble_log_probabilities / 0.7171436548233032
>>>>>>> REPLACE