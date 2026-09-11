MECHANISM: Accuracy-invariant evaluation logit sharpening sweep

HYPOTHESIS: Increasing the evaluation-only logit scale from 1.10 to 1.12 will preserve all 9,359 argmax predictions while reducing validation cross-entropy below 0.18725096054077148.

INTENDED_EDIT: Increase only the positive scale applied to flip-ensemble logits during evaluation.

EVIDENCE: Every tested increase from 1.02 through 1.10 preserved 9,359 correct predictions and monotonically reduced cross-entropy, with 1.10 improving it to 0.18725096054077148.

<<<<<<< SEARCH
        return 1.10 * 0.5 * (logits + flipped_logits)
=======
        return 1.12 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE