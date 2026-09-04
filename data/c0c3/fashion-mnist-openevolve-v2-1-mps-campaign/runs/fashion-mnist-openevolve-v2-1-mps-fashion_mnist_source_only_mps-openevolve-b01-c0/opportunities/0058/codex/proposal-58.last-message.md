MECHANISM: Evidence-fitted decision-preserving temperature calibration

HYPOTHESIS: An ensemble temperature of 0.76 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.80 to the untested near-optimal value 0.76.

EVIDENCE: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy with diminishing gains, placing the apparent calibration minimum near 0.75–0.76; previous nearby attempts timed out without producing contrary performance evidence.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.80
=======
        return ensemble_log_probabilities / 0.76
>>>>>>> REPLACE