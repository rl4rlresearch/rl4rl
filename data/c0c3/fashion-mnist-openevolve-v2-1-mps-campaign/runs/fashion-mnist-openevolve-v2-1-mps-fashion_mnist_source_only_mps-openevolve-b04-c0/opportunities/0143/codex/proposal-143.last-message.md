MECHANISM: Local quadratic vertical-TTA weight refinement

HYPOTHESIS: A 13.3671875% symmetric vertical blend will preserve all 9,359 correct predictions through the existing guards and reduce cross-entropy below 0.18436321144104004.

INTENDED_EDIT: Increase total translated-view weight from 13.296875% to 13.3671875%, assigning 6.68359375% to each vertical shift.

EVIDENCE: Cross-entropy improved from 0.18436353340148925 at 13.15625% to 0.18436321144104004 at 13.296875%, while the 14.2578125% probe was worse; local quadratic interpolation places the estimated minimum near 13.37%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86703125 * ensemble_logits
            + 0.066484375 * down_logits
            + 0.066484375 * up_logits
        )
=======
        translation_refined_logits = (
            0.866328125 * ensemble_logits
            + 0.0668359375 * down_logits
            + 0.0668359375 * up_logits
        )
>>>>>>> REPLACE