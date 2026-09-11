MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.71612548828125 will retain 9,192 correct predictions while lowering cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Bisect the diagonal-view weight between the failed 0.71611328125 and successful 0.7161376953125 settings, and normalize by the exact total ensemble weight of 15.72900390625.

EVIDENCE: Weight 0.7161376953125 achieved 9,192 correct with 0.23742913589477538 cross-entropy, while 0.71611328125 achieved 9,191 correct with slightly lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
            weight = 0.7162109375 if index in diagonal_indices else 1.0
=======
            weight = 0.71612548828125 if index in diagonal_indices else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 15.7296875
=======
        return logit_sum / 15.72900390625
>>>>>>> REPLACE