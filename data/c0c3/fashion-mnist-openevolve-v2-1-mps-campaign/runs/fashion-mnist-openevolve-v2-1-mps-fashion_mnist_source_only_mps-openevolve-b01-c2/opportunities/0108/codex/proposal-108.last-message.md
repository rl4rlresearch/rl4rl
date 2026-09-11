MECHANISM: Verified flip-ensemble temperature calibration

HYPOTHESIS: Scaling evaluation logits from 1.15 to 1.184 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.

INTENDED_EDIT: Increase only the evaluation-time scale applied to the symmetric flip-logit ensemble.

EVIDENCE: Reference Designs 1 and 2 verified this exact change at 9,330 correct and 0.196249866 cross-entropy, the best reported validation score; later verification failures produced no contrary metrics.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE