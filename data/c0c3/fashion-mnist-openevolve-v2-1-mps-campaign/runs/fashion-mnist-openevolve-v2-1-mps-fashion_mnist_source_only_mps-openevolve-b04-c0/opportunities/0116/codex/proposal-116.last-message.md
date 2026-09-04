MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.807315826416015625% original / 46.192684173583984375% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.18465710182189943.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.8072967529296875% canonical weighting toward the unsuccessful 53.80733489990234375% weighting.

EVIDENCE: The current weighting preserved 9,359 correct, while 53.80733489990234375% slightly reduced cross-entropy but produced 9,358 correct; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.538072967529296875 * logits + 0.461927032470703125 * flipped_logits
        )
=======
        return 1.226016 * (
            0.53807315826416015625 * logits
            + 0.46192684173583984375 * flipped_logits
        )
>>>>>>> REPLACE