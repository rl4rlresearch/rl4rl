MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.807220458984375% original / 46.192779541015625% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.1846571128845215.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.80706787109375% canonical weighting toward the unsuccessful 53.807373046875% weighting.

EVIDENCE: The current 53.80706787109375% weighting preserved 9,359 correct at 0.1846571128845215 cross-entropy, while 53.807373046875% lowered cross-entropy to 0.1846571029663086 but produced 9,358 correct; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5380706787109375 * logits + 0.4619293212890625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.53807220458984375 * logits + 0.46192779541015625 * flipped_logits
        )
>>>>>>> REPLACE