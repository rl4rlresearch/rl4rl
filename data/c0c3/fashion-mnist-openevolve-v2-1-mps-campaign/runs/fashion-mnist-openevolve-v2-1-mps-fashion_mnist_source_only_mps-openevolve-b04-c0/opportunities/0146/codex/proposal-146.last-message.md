MECHANISM: Midpoint localization of the guarded vertical-TTA boundary

HYPOTHESIS: A 13.40234375% symmetric vertical blend will preserve all 9,359 correct predictions and reduce cross-entropy below 0.18436306648254394 without triggering the degradation observed at 13.4375%.

INTENDED_EDIT: Increase total translated-view weight from 13.3671875% to 13.40234375%, assigning 6.701171875% to each vertical shift.

EVIDENCE: Cross-entropy improved through 13.3671875% but sharply worsened at 13.4375%, suggesting a discrete confidence-gating boundary; testing their midpoint most directly localizes that boundary while probing the still-improving direction.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.866328125 * ensemble_logits
            + 0.0668359375 * down_logits
            + 0.0668359375 * up_logits
        )
=======
        translation_refined_logits = (
            0.8659765625 * ensemble_logits
            + 0.06701171875 * down_logits
            + 0.06701171875 * up_logits
        )
>>>>>>> REPLACE