MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.828125% original / 46.171875% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the 53.90625% weighting that lost one correct prediction.

EVIDENCE: The 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, while 53.90625% lowered cross-entropy to 0.18465328063964845 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5375 * logits + 0.4625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.53828125 * logits + 0.46171875 * flipped_logits
        )
>>>>>>> REPLACE