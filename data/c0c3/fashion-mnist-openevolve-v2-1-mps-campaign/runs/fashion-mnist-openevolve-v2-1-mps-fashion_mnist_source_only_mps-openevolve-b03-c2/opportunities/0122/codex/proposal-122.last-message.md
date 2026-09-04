MECHANISM: Fine-grained evaluation temperature calibration

HYPOTHESIS: Raising the evaluation-only multiplier to 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.21445223388671875.

INTENDED_EDIT: Change only the ensemble logit multiplier from 1.356 to 1.35605.

EVIDENCE: Reference Design 3 verified the otherwise-identical implementation at 9,256 correct and 0.2144522300720215 cross-entropy; prior verification failures supplied no contradictory measurements.

<<<<<<< SEARCH
        return 1.356 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE