MECHANISM: Conservative evaluation-temperature sharpening

HYPOTHESIS: Scaling ensemble logits to 1.175 will preserve exactly 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.15 to 1.175.

EVIDENCE: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while cross-entropy decreased from 0.200571 to 0.198018 to 0.196639; 1.175 cautiously approaches the fitted optimum near 1.184, and prior higher-scale timeouts provide no negative validation evidence.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.175 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE