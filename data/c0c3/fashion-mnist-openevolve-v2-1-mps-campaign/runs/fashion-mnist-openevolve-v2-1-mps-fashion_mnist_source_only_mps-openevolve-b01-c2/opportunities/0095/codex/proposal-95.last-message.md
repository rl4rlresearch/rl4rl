MECHANISM: Verified flip-ensemble temperature calibration

HYPOTHESIS: Scaling evaluation logits from 1.15 to 1.184 will preserve exactly 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.

INTENDED_EDIT: Increase only the positive evaluation-time scale applied to the symmetric flip-ensemble logits.

EVIDENCE: Reference Designs 2 and 3 independently achieved the best reported validation score with this exact change; recent verification failures produced no contrary metric evidence.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE