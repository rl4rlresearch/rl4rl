MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 54.375% original / 45.625% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the lower-cross-entropy 55% weighting, retaining the established calibration scale.

EVIDENCE: The 53.75% weighting preserved 9,359 correct with 0.1846593635559082 cross-entropy, while 55% lowered cross-entropy to 0.18461807746887207 but lost one correct prediction; their midpoint is the most informative probe of the accuracy boundary.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5375 * logits + 0.4625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.54375 * logits + 0.45625 * flipped_logits
        )
>>>>>>> REPLACE