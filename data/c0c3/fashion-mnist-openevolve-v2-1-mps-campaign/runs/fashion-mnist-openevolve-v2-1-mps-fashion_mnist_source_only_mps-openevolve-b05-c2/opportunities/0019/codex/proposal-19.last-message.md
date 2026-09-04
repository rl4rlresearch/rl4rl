MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.71611328125 will retain 9,192 correct predictions while lowering cross-entropy below 0.2374291809082031.

INTENDED_EDIT: Bisect the diagonal-view weight between 0.716015625 and 0.7162109375 and adjust ensemble normalization to 15.72890625.

EVIDENCE: Weight 0.7162109375 achieved 9,192 correct, while 0.716015625 achieved 9,191 with slightly lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
                weight = 0.71875 if is_diagonal else 1.0
=======
                weight = 0.71611328125 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.75
=======
        return logit_sum / 15.72890625
>>>>>>> REPLACE