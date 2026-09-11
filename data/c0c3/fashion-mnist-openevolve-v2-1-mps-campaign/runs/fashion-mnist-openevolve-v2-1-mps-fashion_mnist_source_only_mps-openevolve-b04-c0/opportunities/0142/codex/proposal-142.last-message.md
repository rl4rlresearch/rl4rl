MECHANISM: Retry of quadratic-interpolated vertical-TTA optimum

HYPOTHESIS: A 13.296875% symmetric vertical blend will preserve all 9,359 correct predictions through the existing guards and reduce cross-entropy below 0.18436353340148925.

INTENDED_EDIT: Increase total translated-view weight from 13.15625% to 13.296875%, assigning 6.6484375% to each vertical shift.

EVIDENCE: The measured 13.0625%, 13.15625%, and 14.2578125% probes estimate an optimum near 13.30%; the previous 13.296875% verification timed out without producing contrary metric evidence.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8684375 * ensemble_logits
            + 0.06578125 * down_logits
            + 0.06578125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86703125 * ensemble_logits
            + 0.066484375 * down_logits
            + 0.066484375 * up_logits
        )
>>>>>>> REPLACE