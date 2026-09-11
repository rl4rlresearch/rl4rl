MECHANISM: Minimum-effective flip-inconsistency weighting

HYPOTHESIS: Reducing the flip-inconsistency penalty from 15% to 12.5% will retain 9,289 correct predictions while lowering validation cross-entropy below 0.2067018039703369.

INTENDED_EDIT: Change inconsistent offset-pair weights from 0.85 to 0.875, leaving training, center preference, and agreement calibration unchanged.

EVIDENCE: The 15% and 25% penalties both produced 9,289 correct, but increasing the penalty worsened cross-entropy from 0.2067018039703369 to 0.20672024116516113; this motivates testing a milder penalty for the same accuracy gain with better calibration.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
=======
        offset_weights = base_weights * (0.875 + 0.125 * flip_consistency)
>>>>>>> REPLACE