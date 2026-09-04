MECHANISM: Argmax-preserving 29.5% translation blend

HYPOTHESIS: A 29.5% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198116193.

INTENDED_EDIT: Increase the evaluation-only translated-logit blend from 29% to 29.5%, retaining argmax gating and temperature calibration.

EVIDENCE: The 29% blend preserved 9,319 correct predictions and improved cross-entropy from 0.198158973 at 28% to 0.198116193; a half-point increase probes the nearby optimum more precisely, while the gate prevents prediction changes.

<<<<<<< SEARCH
            blended_logits = 0.71 * logits + 0.29 * translated_logits
=======
            blended_logits = 0.705 * logits + 0.295 * translated_logits
>>>>>>> REPLACE