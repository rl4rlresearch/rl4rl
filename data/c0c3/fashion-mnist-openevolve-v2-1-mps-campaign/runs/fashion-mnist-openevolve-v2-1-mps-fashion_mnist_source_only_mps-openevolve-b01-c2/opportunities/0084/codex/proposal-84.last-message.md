MECHANISM: Verified flip-ensemble temperature calibration

HYPOTHESIS: Scaling evaluation logits from 1.15 to 1.184 will preserve all 9,330 correct predictions while reducing validation cross-entropy to approximately 0.196250.

INTENDED_EDIT: Increase only the positive evaluation-time scale applied to symmetric flip-ensemble logits.

EVIDENCE: Reference Designs 1 and 2 verified this exact scale at 9,330 correct and 0.196249866 cross-entropy, strictly improving the current design’s tie-breaker without changing training cost, parameters, or argmax predictions.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE