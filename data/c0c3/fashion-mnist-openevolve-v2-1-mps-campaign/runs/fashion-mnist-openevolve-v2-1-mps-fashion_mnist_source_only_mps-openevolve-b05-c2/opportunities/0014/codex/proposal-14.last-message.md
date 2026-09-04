MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.715625 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23743045 achieved at 0.71875.

INTENDED_EDIT: Reduce diagonal-view weight from 0.75 to 0.715625 and adjust ensemble normalization from 16.0 to 15.725.

EVIDENCE: Weight 0.71875 achieved 9,192 correct, while 0.7125 achieved 9,191 correct with lower cross-entropy; their midpoint directly probes the lowest weight preserving the leading correct count.

<<<<<<< SEARCH
                weight = 0.75 if is_diagonal else 1.0
=======
                weight = 0.715625 if is_diagonal else 1.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logit_sum / 16.0
=======
        return logit_sum / 15.725
>>>>>>> REPLACE