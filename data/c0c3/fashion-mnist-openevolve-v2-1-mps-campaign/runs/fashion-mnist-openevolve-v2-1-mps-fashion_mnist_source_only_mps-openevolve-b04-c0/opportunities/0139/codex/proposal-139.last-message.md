MECHANISM: Local quadratic vertical-TTA weight refinement

HYPOTHESIS: A 13.0625% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.1843650894165039.

INTENDED_EDIT: Increase total translated-view weight from 12.625% to 13.0625%, assigning 6.53125% to each vertical shift.

EVIDENCE: The 12.625% blend improved cross-entropy over 12.5%, while 14.2578125% was worse; quadratic interpolation of these probes estimates a local minimum near 13.06%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.87375 * ensemble_logits
            + 0.063125 * down_logits
            + 0.063125 * up_logits
        )
=======
        translation_refined_logits = (
            0.869375 * ensemble_logits
            + 0.0653125 * down_logits
            + 0.0653125 * up_logits
        )
>>>>>>> REPLACE