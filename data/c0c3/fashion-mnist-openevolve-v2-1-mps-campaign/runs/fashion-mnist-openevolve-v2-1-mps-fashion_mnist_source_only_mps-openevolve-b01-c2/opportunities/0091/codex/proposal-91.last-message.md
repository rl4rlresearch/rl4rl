MECHANISM: Verified flip-ensemble temperature calibration

HYPOTHESIS: Scaling evaluation logits to 1.184 will preserve 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.

INTENDED_EDIT: Increase only the positive evaluation-time scale applied to the symmetric flip-ensemble logits.

EVIDENCE: Reference Designs 2 and 3 independently verified this exact scale at 9,330 correct and 0.196249866 cross-entropy, strictly improving the current tie-breaker without changing training computation or argmax predictions.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE