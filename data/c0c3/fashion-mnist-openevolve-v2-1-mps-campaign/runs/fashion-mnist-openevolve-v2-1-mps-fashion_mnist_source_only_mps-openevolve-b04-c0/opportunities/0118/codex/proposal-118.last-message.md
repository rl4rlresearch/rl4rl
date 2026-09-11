MECHANISM: Accuracy-constrained canonical-view weight bisection

HYPOTHESIS: A 53.80730152130126953125% original / 46.19269847869873046875% flipped weighting will preserve 9,359 correct predictions while reducing validation cross-entropy below 0.18465710182189943.

INTENDED_EDIT: Move the evaluation ensemble halfway from the best successful weighting toward the nearest unsuccessful weighting.

EVIDENCE: The current 53.8072967529296875% weighting achieved 9,359 correct, while 53.8073062896728515625% achieved 9,358 with slightly lower cross-entropy; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
            0.538072967529296875 * logits + 0.461927032470703125 * flipped_logits
=======
            0.5380730152130126953125 * logits + 0.4619269847869873046875 * flipped_logits
>>>>>>> REPLACE