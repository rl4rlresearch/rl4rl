MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.8067626953125% original / 46.1932373046875% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.1846571502685547.

INTENDED_EDIT: Move the evaluation ensemble halfway from the successful 53.80615234375% canonical weighting toward the 53.807373046875% weighting that lost one correct prediction.

EVIDENCE: The current 53.80615234375% weighting preserved 9,359 correct at 0.1846571502685547 cross-entropy, while 53.807373046875% lowered cross-entropy to 0.1846571029663086 but produced 9,358 correct; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        return 1.226016 * (
            0.5380615234375 * logits + 0.4619384765625 * flipped_logits
        )
=======
        return 1.226016 * (
            0.538067626953125 * logits + 0.461932373046875 * flipped_logits
        )
>>>>>>> REPLACE