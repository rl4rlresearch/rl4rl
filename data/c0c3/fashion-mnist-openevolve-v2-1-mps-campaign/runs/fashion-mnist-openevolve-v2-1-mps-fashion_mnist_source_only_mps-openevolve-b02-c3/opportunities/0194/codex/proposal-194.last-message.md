MECHANISM: Verified crop-power restoration

HYPOTHESIS: Setting the crop fusion power to 1.3515 will preserve 9,349 correct predictions and reduce validation cross-entropy from 0.1876555145263672 to approximately 0.1876555103302002.

INTENDED_EDIT: Replace the current 1.3477 crop-consensus power with the highest-scoring verified value, preserving all other training, fusion, and calibration behavior.

EVIDENCE: Reference Design 1 used fusion power 1.3515 with otherwise identical code and achieved the best available validation_score, while the nearby 1.3509 result was slightly worse.

<<<<<<< SEARCH
        fusion_power = 1.3477
=======
        fusion_power = 1.3515
>>>>>>> REPLACE