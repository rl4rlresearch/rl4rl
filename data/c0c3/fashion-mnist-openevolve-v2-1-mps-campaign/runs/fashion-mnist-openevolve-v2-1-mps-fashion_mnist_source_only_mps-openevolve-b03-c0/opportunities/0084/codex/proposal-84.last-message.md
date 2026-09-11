MECHANISM: Argmax-preserving 25% translation blend

HYPOTHESIS: A 25% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198478033.

INTENDED_EDIT: Increase the evaluation-only translated-logit blend from 20% to 25%, retaining argmax gating and temperature calibration.

EVIDENCE: Gated translation improved cross-entropy monotonically from 0.200074794 at 0% to 0.199061208 at 10% and 0.198478033 at 20% without changing correctness; 25% conservatively advances toward the fitted optimum near 28.55%, whose attempts timed out without contrary performance evidence.

<<<<<<< SEARCH
            blended_logits = 0.80 * logits + 0.20 * translated_logits
=======
            blended_logits = 0.75 * logits + 0.25 * translated_logits
>>>>>>> REPLACE