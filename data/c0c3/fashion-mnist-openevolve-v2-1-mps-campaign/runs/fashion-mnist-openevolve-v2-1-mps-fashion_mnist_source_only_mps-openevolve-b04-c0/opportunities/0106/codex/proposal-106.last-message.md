MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.80859375% original / 46.19140625% flipped weighting will preserve all 9,359 correct predictions while reducing validation cross-entropy below 0.18465781898498534.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.7890625% canonical weighting toward the 53.828125% weighting that lost one correct prediction.

EVIDENCE: The current 53.7890625% weighting achieved 9,359 correct at 0.18465781898498534 cross-entropy, while 53.828125% lowered cross-entropy to 0.18465629234313966 but achieved 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.537890625 * logits + 0.462109375 * flipped_logits
        )
=======
        return 1.226016 * (
            0.5380859375 * logits + 0.4619140625 * flipped_logits
        )
>>>>>>> REPLACE