MECHANISM: Verified symmetric-ensemble temperature sharpening

HYPOTHESIS: Scaling evaluation logits from 1.05 to 1.15 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196639.

INTENDED_EDIT: Change only the evaluation-time symmetric flip-ensemble logit scale to the best verified value.

EVIDENCE: Reference Design 1 differs from the current design only in this scale and achieved the same 9,330 correct predictions with lower cross-entropy, improving validation_score from 9330.416468 to 9330.417837; later timeout results provide no contrary validation evidence.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE