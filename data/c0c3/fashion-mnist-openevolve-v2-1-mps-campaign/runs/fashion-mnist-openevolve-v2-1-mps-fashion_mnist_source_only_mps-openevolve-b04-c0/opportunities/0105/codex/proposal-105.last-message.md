MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.7890625% original / 46.2109375% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.1846593635559082.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.75% canonical weighting toward the 53.828125% weighting that lost one correct prediction.

EVIDENCE: The current 53.75% weighting achieved 9,359 correct at 0.1846593635559082 cross-entropy, while 53.828125% lowered cross-entropy to 0.18465629234313966 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5375 * logits + 0.4625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.537890625 * logits + 0.462109375 * flipped_logits
        )
>>>>>>> REPLACE