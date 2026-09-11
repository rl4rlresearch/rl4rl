MECHANISM: Stronger confidence-gated vertical-translation TTA

HYPOTHESIS: Increasing the symmetric vertical-translation blend from 3.125% to 6.25% will preserve all 9,359 predictions through the existing argmax guard while reducing validation cross-entropy below 0.1844868465423584.

INTENDED_EDIT: Double each translated-view weight from 1.5625% to 3.125%, reducing the canonical ensemble weight accordingly.

EVIDENCE: The guarded 3.125% vertical blend preserved 9,359 correct predictions and improved cross-entropy from 0.18456672592163087 to 0.1844868465423584, supporting a stronger probe in the same direction.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.96875 * ensemble_logits
            + 0.015625 * down_logits
            + 0.015625 * up_logits
        )
=======
        translation_refined_logits = (
            0.9375 * ensemble_logits
            + 0.03125 * down_logits
            + 0.03125 * up_logits
        )
>>>>>>> REPLACE