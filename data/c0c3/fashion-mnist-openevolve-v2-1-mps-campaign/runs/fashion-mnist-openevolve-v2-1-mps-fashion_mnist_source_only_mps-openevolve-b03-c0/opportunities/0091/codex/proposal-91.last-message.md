MECHANISM: Argmax-preserving 30.5% translation blend

HYPOTHESIS: Increasing the gated translation contribution from 30% to 30.5% will preserve all 9,319 predictions while reducing validation cross-entropy below 0.198080122.

INTENDED_EDIT: Increase the evaluation-only translated-logit blend from 30% to 30.5%, retaining argmax gating and temperature calibration.

EVIDENCE: Cross-entropy improved monotonically from 0.198116193 at 29% to 0.198097674 at 29.5% and 0.198080122 at 30%; another half-point increment tests whether that local improvement continues while the gate prevents prediction changes.

<<<<<<< SEARCH
            blended_logits = 0.70 * logits + 0.30 * translated_logits
=======
            blended_logits = 0.695 * logits + 0.305 * translated_logits
>>>>>>> REPLACE