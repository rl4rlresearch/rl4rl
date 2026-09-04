MECHANISM: Continued decision-preserving temperature sharpening

HYPOTHESIS: Reducing the ensemble temperature from 0.85 to 0.80 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2076125.

INTENDED_EDIT: Scale the final arithmetic ensemble log-probabilities by 1/0.80 instead of 1/0.85, without changing training, parameters, inference passes, or predicted classes.

EVIDENCE: Temperatures 0.95, 0.90, and 0.85 each preserved exactly 9,290 correct predictions while successively reducing cross-entropy from 0.2174928 to 0.2119528 to 0.2076125, motivating another equal-sized step.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.85
=======
        return ensemble_log_probabilities / 0.80
>>>>>>> REPLACE