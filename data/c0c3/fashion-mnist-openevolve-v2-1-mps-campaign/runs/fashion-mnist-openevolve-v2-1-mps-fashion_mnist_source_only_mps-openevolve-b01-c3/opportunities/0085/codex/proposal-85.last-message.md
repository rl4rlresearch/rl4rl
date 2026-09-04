MECHANISM: Intermediate center-biased full-view pooling

HYPOTHESIS: Raising the center-offset weight from 1.5× to 1.75× will exceed 9,287 correct predictions by strengthening validation-aligned evidence while retaining all shifted and flipped views.

INTENDED_EDIT: Increase only the unshifted offset’s pooling weight and adjust the normalization accordingly; preserve training and the 1.225 logit scale.

EVIDENCE: Moving from uniform pooling to 1.5× center weighting preserved 9,287 correct while lowering cross-entropy from 0.2067398 to 0.2067078; the 2× attempt timed out, making 1.75× the informative untested midpoint.

<<<<<<< SEARCH
        pooled_logits = (
            1.5 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5
=======
        pooled_logits = (
            1.75 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.75
>>>>>>> REPLACE