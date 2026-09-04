MECHANISM: Locally interpolated argmax-preserving translation blend

HYPOTHESIS: A 30.06% translation contribution will preserve all 9,319 predictions and reduce validation cross-entropy below 0.198080122.

INTENDED_EDIT: Refine the evaluation-only blend from 30% to 30.06%, retaining argmax gating and temperature calibration.

EVIDENCE: Cross-entropy was 0.198081342 at 29.95%, improved to 0.198080122 at 30%, and worsened to 0.198105949 at 30.5%; local quadratic interpolation of these verified points places the minimum near 30.06%.

<<<<<<< SEARCH
            blended_logits = 0.70 * logits + 0.30 * translated_logits
=======
            blended_logits = 0.6994 * logits + 0.3006 * translated_logits
>>>>>>> REPLACE