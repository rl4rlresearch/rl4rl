MECHANISM: Local quadratic vertical-TTA interpolation

HYPOTHESIS: A 13.15625% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436376991271972.

INTENDED_EDIT: Increase total translated-view weight from 13.0625% to 13.15625%, assigning 6.578125% to each vertical shift.

EVIDENCE: Cross-entropy improved at 12.625% and 13.0625% but worsened at 14.2578125%; quadratic interpolation of these three nearby probes estimates a minimum near 13.16%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.869375 * ensemble_logits
            + 0.0653125 * down_logits
            + 0.0653125 * up_logits
        )
=======
        translation_refined_logits = (
            0.8684375 * ensemble_logits
            + 0.06578125 * down_logits
            + 0.06578125 * up_logits
        )
>>>>>>> REPLACE