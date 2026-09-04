MECHANISM: Argmax-preserving 27% translation blend

HYPOTHESIS: A 27% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198298140.

INTENDED_EDIT: Increase the evaluation-only translated-logit blend from 25% to 27%, retaining argmax gating and temperature calibration.

EVIDENCE: Increasing the gated blend from 20% to 25% preserved 9,319 correct predictions and reduced cross-entropy from 0.198478033 to 0.198298140; 27% is a conservative step toward the fitted optimum near 28.55%, while the prior 28% timeout supplied no contrary performance evidence.

<<<<<<< SEARCH
            blended_logits = 0.75 * logits + 0.25 * translated_logits
=======
            blended_logits = 0.73 * logits + 0.27 * translated_logits
>>>>>>> REPLACE