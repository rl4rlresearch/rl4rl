MECHANISM: Evidence-fitted decision-preserving temperature calibration

HYPOTHESIS: An ensemble temperature of 0.70 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2045404.

INTENDED_EDIT: Scale the final arithmetic-ensemble log-probabilities by 1/0.70 instead of 1/0.80.

EVIDENCE: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while cross-entropy decreased with diminishing gains; the measured curve estimates its minimum near 0.70, and the prior 0.70 run timed out without contradicting that hypothesis.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.80
=======
        return ensemble_log_probabilities / 0.70
>>>>>>> REPLACE