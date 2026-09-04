MECHANISM: Lower-side micro-bracketing temperature calibration

HYPOTHESIS: A temperature of 0.717143 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924331665039.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.71714375 to the adjacent lower-side probe 0.717143.

EVIDENCE: Reducing the temperature from 0.71715 to 0.71714375 preserved all predictions and lowered cross-entropy by 7.63e-10; a smaller continuation in that direction is the most informative local refinement.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.71714375
=======
        return ensemble_log_probabilities / 0.717143
>>>>>>> REPLACE