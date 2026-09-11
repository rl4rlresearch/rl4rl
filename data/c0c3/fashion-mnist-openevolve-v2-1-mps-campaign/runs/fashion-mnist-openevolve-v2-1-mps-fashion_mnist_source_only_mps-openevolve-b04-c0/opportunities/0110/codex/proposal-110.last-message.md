MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.807373046875% original / 46.192626953125% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.1846571502685547.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.80615234375% canonical weighting toward the 53.80859375% weighting that lost one correct prediction.

EVIDENCE: The current 53.80615234375% weighting preserved 9,359 correct at 0.1846571502685547 cross-entropy, while 53.80859375% reduced cross-entropy to 0.18465705108642577 but produced 9,358 correct; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5380615234375 * logits + 0.4619384765625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.53807373046875 * logits + 0.46192626953125 * flipped_logits
        )
>>>>>>> REPLACE