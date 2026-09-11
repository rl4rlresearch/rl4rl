MECHANISM: Verified flip-ensemble temperature calibration

HYPOTHESIS: Scaling evaluation logits to 1.184 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.196639 to approximately 0.196250.

INTENDED_EDIT: Increase only the evaluation-time scale applied to the symmetric flip-ensemble logits.

EVIDENCE: Reference Designs 1 and 2 verified this exact implementation at 9,330 correct and 0.196249866 cross-entropy, the highest available validation score; subsequent verification non-completions provide no contrary metric evidence.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE