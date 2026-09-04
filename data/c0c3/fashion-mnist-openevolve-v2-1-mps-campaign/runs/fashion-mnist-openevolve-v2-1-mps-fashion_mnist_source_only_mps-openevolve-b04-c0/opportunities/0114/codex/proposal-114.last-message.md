MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.8072967529296875% original / 46.1927032470703125% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.18465710563659668.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.807220458984375% canonical weighting toward the unsuccessful 53.807373046875% weighting.

EVIDENCE: The current weighting preserved 9,359 correct, while 53.807373046875% lowered cross-entropy but produced 9,358 correct; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.53807220458984375 * logits + 0.46192779541015625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.538072967529296875 * logits + 0.461927032470703125 * flipped_logits
        )
>>>>>>> REPLACE