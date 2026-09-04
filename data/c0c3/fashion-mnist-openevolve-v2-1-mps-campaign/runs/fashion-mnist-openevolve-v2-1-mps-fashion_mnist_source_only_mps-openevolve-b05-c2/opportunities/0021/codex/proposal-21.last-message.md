MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.7161376953125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742915229797362.

INTENDED_EDIT: Bisect the diagonal-view weight between 0.71611328125 and 0.716162109375 and normalize by the exact total ensemble weight, 15.7291015625.

EVIDENCE: Weight 0.716162109375 achieved 9,192 correct, while 0.71611328125 achieved 9,191 with lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
                weight = 0.716162109375 if is_diagonal else 1.0
=======
                weight = 0.7161376953125 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.729296875
=======
        return logit_sum / 15.7291015625
>>>>>>> REPLACE