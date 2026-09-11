MECHANISM: Argmax-preserving 30% translation blend

HYPOTHESIS: A 30% gated translation contribution will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198478033.

INTENDED_EDIT: Increase the evaluation-only translated-logit blend from 20% to 30%, retaining argmax gating and temperature calibration.

EVIDENCE: Cross-entropy improved monotonically from 0.200074794 at 0% translation to 0.199061208 at 10% and 0.198478033 at 20%; the fitted optimum is near 28.55%, while two 28.55% attempts timed out without contrary performance evidence.

<<<<<<< SEARCH
            blended_logits = 0.80 * logits + 0.20 * translated_logits
=======
            blended_logits = 0.70 * logits + 0.30 * translated_logits
>>>>>>> REPLACE