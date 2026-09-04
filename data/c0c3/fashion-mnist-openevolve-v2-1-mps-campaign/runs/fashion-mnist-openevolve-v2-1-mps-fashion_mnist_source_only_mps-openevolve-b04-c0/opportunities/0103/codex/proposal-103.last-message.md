MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.90625% original / 46.09375% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the unsuccessful 54.0625% weighting, retaining the established calibration and training procedure.

EVIDENCE: The 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, whereas 54.0625% reduced cross-entropy but fell to 9,357 correct; their midpoint is the most informative remaining probe of this accuracy boundary.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5375 * logits + 0.4625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.5390625 * logits + 0.4609375 * flipped_logits
        )
>>>>>>> REPLACE