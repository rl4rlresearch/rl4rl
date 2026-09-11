MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5425% to 13.545% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843627410888672.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.77125% to 6.7725%.

EVIDENCE: Every tested increase from 13.4375% through 13.5425% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.864575 * ensemble_logits
            + 0.0677125 * down_logits
            + 0.0677125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86455 * ensemble_logits
            + 0.067725 * down_logits
            + 0.067725 * up_logits
        )
>>>>>>> REPLACE