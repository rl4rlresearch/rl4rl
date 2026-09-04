MECHANISM: Verified evaluation-temperature calibration

HYPOTHESIS: Scaling evaluation logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.21445224533081056 to 0.2144522300720215.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.35641 to 1.35605.

EVIDENCE: Reference Design 2 is otherwise identical and achieved the same 9,256 correct predictions with the best verified cross-entropy and a strictly higher validation_score than the current design.

<<<<<<< SEARCH
        return 1.35641 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE