MECHANISM: Stronger confidence-gated vertical-translation TTA

HYPOTHESIS: Increasing the symmetric vertical-translation blend from 12.5% to 25% will preserve all 9,359 predictions through the existing argmax guard and reduce validation cross-entropy below 0.18436553497314453.

INTENDED_EDIT: Double each translated-view weight from 6.25% to 12.5%, reducing the horizontal-ensemble weight from 87.5% to 75%.

EVIDENCE: Doubling the guarded vertical blend from 6.25% to 12.5% preserved 9,359 correct predictions and improved cross-entropy from 0.18442666282653808 to 0.18436553497314453, with no observed saturation.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.875 * ensemble_logits
            + 0.0625 * down_logits
            + 0.0625 * up_logits
        )
=======
        translation_refined_logits = (
            0.75 * ensemble_logits
            + 0.125 * down_logits
            + 0.125 * up_logits
        )
>>>>>>> REPLACE