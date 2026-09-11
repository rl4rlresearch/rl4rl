MECHANISM: Local quadratic vertical-TTA weight refinement

HYPOTHESIS: A 13.4375% symmetric vertical blend will preserve all 9,359 correct predictions through the existing guards and reduce cross-entropy below 0.18436306648254394.

INTENDED_EDIT: Increase total translated-view weight from 13.3671875% to 13.4375%, assigning 6.71875% to each vertical shift.

EVIDENCE: Cross-entropy improved at 13.296875% and 13.3671875% but worsened at 14.2578125%; quadratic interpolation of these probes estimates a minimum near 13.44%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.866328125 * ensemble_logits
            + 0.0668359375 * down_logits
            + 0.0668359375 * up_logits
        )
=======
        translation_refined_logits = (
            0.865625 * ensemble_logits
            + 0.0671875 * down_logits
            + 0.0671875 * up_logits
        )
>>>>>>> REPLACE