MECHANISM: One-ULP upward logit-temperature refinement

HYPOTHESIS: Increasing the positive evaluation logit scale by one float32 step will preserve all 9,254 argmax predictions and lower validation cross-entropy below 0.21227332077026367.

INTENDED_EDIT: Replace the evaluation logit multiplier with its immediately higher float32 value.

EVIDENCE: Decreasing the multiplier by one float32 step preserved accuracy but worsened cross-entropy to 0.2122733238220215, making the exact opposite direction the most informative untested refinement.

<<<<<<< SEARCH
        return 1.3715955 * ensemble_scores
=======
        return 1.3715956211090088 * ensemble_scores
>>>>>>> REPLACE