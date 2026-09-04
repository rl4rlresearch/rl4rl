MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.54% to 13.5425% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436274490356444.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.77% to 6.77125%.

EVIDENCE: Every tested increase from 13.4375% through 13.54% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8646 * ensemble_logits
            + 0.0677 * down_logits
            + 0.0677 * up_logits
        )
=======
        translation_refined_logits = (
            0.864575 * ensemble_logits
            + 0.0677125 * down_logits
            + 0.0677125 * up_logits
        )
>>>>>>> REPLACE