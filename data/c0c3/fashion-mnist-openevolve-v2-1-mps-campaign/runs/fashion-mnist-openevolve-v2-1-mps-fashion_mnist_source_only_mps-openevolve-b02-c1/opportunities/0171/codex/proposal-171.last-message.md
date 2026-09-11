MECHANISM: Moderate center-emphasized geometric TTA

HYPOTHESIS: Raising both centered-view weights from 2.0 to 2.5 will correct more borderline translation-sensitive decisions than it harms, exceeding 9,311 correct predictions.

INTENDED_EDIT: Increase the original and horizontally flipped centered-view weights to 2.5 and renormalize the calibrated ensemble by its new total weight of 13.

EVIDENCE: Probability averaging fell to 9,310 correct, favoring logit aggregation; the stronger center-weighting trial timed out without contradictory metrics, so testing its midpoint is the smallest informative change capable of breaking the 9,311 argmax plateau.

<<<<<<< SEARCH
        ensemble = logits * 2.0
=======
        ensemble = logits * 2.5
>>>>>>> REPLACE

<<<<<<< SEARCH
                flipped_logits = flipped_logits * 2.0
=======
                flipped_logits = flipped_logits * 2.5
>>>>>>> REPLACE

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9350)
=======
        return ensemble / (13.0 * 0.9350)
>>>>>>> REPLACE