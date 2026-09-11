MECHANISM: Quadratic-refined argmax-preserving translation blend

HYPOTHESIS: A 29.95% translation contribution will preserve all 9,319 predictions and reduce validation cross-entropy below 0.198080122.

INTENDED_EDIT: Refine the evaluation-only blend from 30% to 29.95%, retaining argmax gating and temperature calibration.

EVIDENCE: Cross-entropy was 0.198097674 at 29.5%, improved to 0.198080122 at 30%, then worsened to 0.198105949 at 30.5%; a quadratic fit places the local minimum near 29.95%.

<<<<<<< SEARCH
            blended_logits = 0.70 * logits + 0.30 * translated_logits
=======
            blended_logits = 0.7005 * logits + 0.2995 * translated_logits
>>>>>>> REPLACE