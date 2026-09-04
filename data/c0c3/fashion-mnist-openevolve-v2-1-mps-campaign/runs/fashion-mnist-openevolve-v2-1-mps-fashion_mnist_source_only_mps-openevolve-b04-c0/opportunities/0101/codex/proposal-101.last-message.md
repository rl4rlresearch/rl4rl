MECHANISM: Accuracy-constrained canonical-view weight bisection retry

HYPOTHESIS: A 54.375% original / 45.625% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.

INTENDED_EDIT: Increase the evaluation ensemble’s canonical-image weight from 53.75% to 54.375%, retaining the established calibration scale and training procedure.

EVIDENCE: The 53.75% weighting preserved 9,359 correct at 0.1846593635559082 cross-entropy, while 55% lowered cross-entropy but lost one correct prediction. The midpoint trial timed out and therefore provided no subject-level evidence against this most informative boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5375 * logits + 0.4625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.54375 * logits + 0.45625 * flipped_logits
        )
>>>>>>> REPLACE