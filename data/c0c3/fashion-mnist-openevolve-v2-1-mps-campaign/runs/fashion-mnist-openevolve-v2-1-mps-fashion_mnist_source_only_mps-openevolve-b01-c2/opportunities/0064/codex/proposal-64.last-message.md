MECHANISM: Verified symmetric-ensemble temperature sharpening

HYPOTHESIS: Scaling evaluation logits from 1.05 to 1.15 will preserve all 9,330 correct predictions while reducing validation cross-entropy from 0.200571 to approximately 0.196639.

INTENDED_EDIT: Change only the evaluation-time symmetric flip-ensemble logit scale to 1.15.

EVIDENCE: All three qualified reference designs use the otherwise identical implementation and achieved 9,330 correct with 0.196639 cross-entropy at scale 1.15, strictly outperforming the current scale 1.05; later timeouts provide no contrary validation result.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE