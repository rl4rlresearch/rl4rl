MECHANISM: Local quadratic vertical-TTA interpolation

HYPOTHESIS: A 12.625% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436553497314453.

INTENDED_EDIT: Increase total translated-view weight from 12.5% to 12.625%, assigning 6.3125% to each vertical shift.

EVIDENCE: Cross-entropy improved through 12.5% translation weight but worsened at 14.2578125%; quadratic interpolation using those results and the 6.25% probe places the estimated minimum near 12.625%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.875 * ensemble_logits
            + 0.0625 * down_logits
            + 0.0625 * up_logits
        )
=======
        translation_refined_logits = (
            0.87375 * ensemble_logits
            + 0.063125 * down_logits
            + 0.063125 * up_logits
        )
>>>>>>> REPLACE