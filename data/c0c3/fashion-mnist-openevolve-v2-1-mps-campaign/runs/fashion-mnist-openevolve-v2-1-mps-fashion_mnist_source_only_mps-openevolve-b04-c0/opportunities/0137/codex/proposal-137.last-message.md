MECHANISM: Quadratic-interpolated vertical-translation TTA

HYPOTHESIS: A 14.2578125% symmetric vertical blend will preserve all 9,359 predictions through the existing guards and reduce validation cross-entropy below 0.18436553497314453.

INTENDED_EDIT: Increase total translated-view weight from 12.5% to 14.2578125%, assigning 7.12890625% to each vertical shift.

EVIDENCE: Cross-entropy improved from 0.18442666282653808 at 6.25% translation weight to 0.18436553497314453 at 12.5%, but worsened to 0.1844776134490967 at 25%; quadratic interpolation of these three probes estimates a minimum near 14.3%.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.875 * ensemble_logits
            + 0.0625 * down_logits
            + 0.0625 * up_logits
        )
=======
        translation_refined_logits = (
            0.857421875 * ensemble_logits
            + 0.0712890625 * down_logits
            + 0.0712890625 * up_logits
        )
>>>>>>> REPLACE