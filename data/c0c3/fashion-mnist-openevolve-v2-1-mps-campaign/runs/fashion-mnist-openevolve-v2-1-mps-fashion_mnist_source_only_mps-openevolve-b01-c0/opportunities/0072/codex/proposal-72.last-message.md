MECHANISM: Adjacent-float temperature micro-calibration

HYPOTHESIS: A temperature of 0.7171437 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924331665039.

INTENDED_EDIT: Move the ensemble temperature to the adjacent lower float32 value.

EVIDENCE: Reducing temperature from 0.71715 to 0.71714375 preserved all predictions and improved cross-entropy; the broader 0.717143 probe timed out, so the adjacent lower representable value is the most conservative informative refinement.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.71714375
=======
        return ensemble_log_probabilities / 0.7171437
>>>>>>> REPLACE