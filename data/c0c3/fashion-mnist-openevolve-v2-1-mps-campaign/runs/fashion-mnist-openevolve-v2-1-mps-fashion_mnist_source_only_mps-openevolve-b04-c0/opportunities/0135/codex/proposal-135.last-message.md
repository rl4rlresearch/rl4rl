MECHANISM: Stronger confidence-gated vertical-translation TTA

HYPOTHESIS: Increasing the symmetric vertical-translation blend from 6.25% to 12.5% will preserve all 9,359 predictions through the existing argmax guard and reduce validation cross-entropy below 0.18442666282653808.

INTENDED_EDIT: Double each translated-view weight from 3.125% to 6.25%, reducing the horizontal-ensemble weight from 93.75% to 87.5%.

EVIDENCE: Doubling the guarded vertical blend from 3.125% to 6.25% preserved 9,359 correct predictions and improved cross-entropy from 0.1844868465423584 to 0.18442666282653808, supporting another stronger probe along the same direction.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.9375 * ensemble_logits
            + 0.03125 * down_logits
            + 0.03125 * up_logits
        )
=======
        translation_refined_logits = (
            0.875 * ensemble_logits
            + 0.0625 * down_logits
            + 0.0625 * up_logits
        )
>>>>>>> REPLACE