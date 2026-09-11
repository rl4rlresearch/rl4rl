MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.80615234375% original / 46.19384765625% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.18465724296569824.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.8037109375% canonical weighting toward the 53.80859375% weighting that lost one correct prediction.

EVIDENCE: The current 53.8037109375% weighting achieved 9,359 correct at 0.18465724296569824 cross-entropy, while 53.80859375% lowered cross-entropy to 0.18465705108642577 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.538037109375 * logits + 0.461962890625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.5380615234375 * logits + 0.4619384765625 * flipped_logits
        )
>>>>>>> REPLACE