MECHANISM: Stronger decision-preserving temperature sharpening

HYPOTHESIS: Reducing the arithmetic ensemble temperature from 0.95 to 0.90 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2174928.

INTENDED_EDIT: Scale the final ensemble log-probabilities by 1/0.90 instead of 1/0.95, with no training, parameter, or inference-pass changes.

EVIDENCE: Temperature 0.95 preserved 9,290 correct predictions and reduced cross-entropy from 0.2241766 to 0.2174928; because positive scaling cannot change argmax decisions, 0.90 directly tests whether further sharpening improves the tie-break objective. Its earlier verification timed out, so the mechanism remains unevaluated.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.95
=======
        return ensemble_log_probabilities / 0.90
>>>>>>> REPLACE