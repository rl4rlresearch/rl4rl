MECHANISM: Upper-adjacent float32 temperature micro-bracketing

HYPOTHESIS: The next representable float32 temperature above the current value will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924331665039.

INTENDED_EDIT: Replace the fitted ensemble temperature with its exact upper-adjacent float32 value.

EVIDENCE: The current fitted temperature achieved the lowest verified cross-entropy; the lower-adjacent probe timed out without contrary evidence, making the untested upper neighbor the most informative remaining bracket point.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.71714375
=======
        return ensemble_log_probabilities / 0.717143833637237548828125
>>>>>>> REPLACE