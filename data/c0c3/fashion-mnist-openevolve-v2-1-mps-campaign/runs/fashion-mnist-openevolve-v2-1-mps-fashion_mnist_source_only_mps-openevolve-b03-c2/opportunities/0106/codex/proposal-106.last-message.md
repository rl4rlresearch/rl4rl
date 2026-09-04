MECHANISM: Verified fine-grained ensemble temperature calibration

HYPOTHESIS: Scaling ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy below 0.21445223388671875.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.356 to 1.35605.

EVIDENCE: Reference Design 3 directly verified this otherwise-identical implementation at 9,256 correct and 0.2144522300720215 cross-entropy, a strict validation_score improvement over the current design.

<<<<<<< SEARCH
        return 1.356 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE