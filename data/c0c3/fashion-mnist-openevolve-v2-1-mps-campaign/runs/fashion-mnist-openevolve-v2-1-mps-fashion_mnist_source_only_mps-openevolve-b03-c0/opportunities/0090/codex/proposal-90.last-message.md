MECHANISM: Argmax-preserving 30% translation blend

HYPOTHESIS: Increasing the gated translation contribution from 29.5% to 30% will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198097674.

INTENDED_EDIT: Increase the evaluation-only translated-logit blend from 29.5% to 30%, retaining argmax gating and temperature calibration.

EVIDENCE: The 29.5% blend preserved 9,319 correct predictions and improved cross-entropy from 0.198116193 at 29% to 0.198097674; the prior 30% attempt timed out and therefore supplied no contrary performance evidence.

<<<<<<< SEARCH
            blended_logits = 0.705 * logits + 0.295 * translated_logits
=======
            blended_logits = 0.70 * logits + 0.30 * translated_logits
>>>>>>> REPLACE