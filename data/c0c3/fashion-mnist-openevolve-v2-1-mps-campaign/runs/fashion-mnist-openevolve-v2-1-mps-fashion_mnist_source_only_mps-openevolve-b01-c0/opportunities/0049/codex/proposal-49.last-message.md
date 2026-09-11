MECHANISM: Further decision-preserving temperature sharpening

HYPOTHESIS: Reducing the arithmetic ensemble temperature from 0.90 to 0.85 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2119528.

INTENDED_EDIT: Scale the final ensemble log-probabilities by 1/0.85 instead of 1/0.90, without changing training, parameters, inference passes, or predicted classes.

EVIDENCE: Successive reductions from temperature 1.00 to 0.95 and then 0.90 preserved exactly 9,290 correct predictions while lowering cross-entropy from 0.2241766 to 0.2174928 and then 0.2119528, motivating one further equal-sized step.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.90
=======
        return ensemble_log_probabilities / 0.85
>>>>>>> REPLACE