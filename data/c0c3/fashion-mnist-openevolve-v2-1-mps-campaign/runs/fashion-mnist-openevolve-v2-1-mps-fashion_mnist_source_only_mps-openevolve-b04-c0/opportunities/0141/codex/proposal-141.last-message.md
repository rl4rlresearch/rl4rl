MECHANISM: Quadratic-interpolated vertical-TTA weight refinement

HYPOTHESIS: A 13.296875% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436353340148925.

INTENDED_EDIT: Increase total translated-view weight from 13.15625% to 13.296875%, assigning 6.6484375% to each vertical shift.

EVIDENCE: Cross-entropy improved from 0.18436376991271972 at 13.0625% to 0.18436353340148925 at 13.15625%, but worsened to 0.1843695240020752 at 14.2578125%; quadratic interpolation of these three probes estimates a minimum near 13.30%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8684375 * ensemble_logits
            + 0.06578125 * down_logits
            + 0.06578125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86703125 * ensemble_logits
            + 0.066484375 * down_logits
            + 0.066484375 * up_logits
        )
>>>>>>> REPLACE