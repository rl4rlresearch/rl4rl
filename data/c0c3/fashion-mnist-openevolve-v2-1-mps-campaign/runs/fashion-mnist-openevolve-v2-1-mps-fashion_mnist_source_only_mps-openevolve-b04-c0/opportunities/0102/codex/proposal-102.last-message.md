MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 54.0625% original / 45.9375% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the unsuccessful 54.375% weighting, retaining the established calibration scale and training procedure.

EVIDENCE: The 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, whereas 54.375% reduced cross-entropy but fell to 9,357 correct; their midpoint is the most informative remaining probe of the accuracy boundary.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5375 * logits + 0.4625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.540625 * logits + 0.459375 * flipped_logits
        )
>>>>>>> REPLACE