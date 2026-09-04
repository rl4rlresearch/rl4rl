MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.53% to 13.5325% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627639770508.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.765% to 6.76625%.

EVIDENCE: Every tested increase from 13.4375% through 13.53% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8647 * ensemble_logits
            + 0.06765 * down_logits
            + 0.06765 * up_logits
        )
=======
        translation_refined_logits = (
            0.864675 * ensemble_logits
            + 0.0676625 * down_logits
            + 0.0676625 * up_logits
        )
>>>>>>> REPLACE