MECHANISM: Verified flip-ensemble temperature calibration

HYPOTHESIS: Scaling evaluation logits by 1.184 will preserve all 9,330 argmax predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196250.

INTENDED_EDIT: Change only the positive evaluation-time scale applied to the symmetric flip-ensemble logits.

EVIDENCE: Reference Design 2 verified this exact implementation at 9,330 correct and 0.196249866 cross-entropy, the best available score and strictly better than the current 1.05 scale at identical correctness.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE