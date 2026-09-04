MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.7171875 will retain 9,192 correct predictions while lowering cross-entropy below the 0.23743045 achieved at 0.71875.

INTENDED_EDIT: Increase diagonal-view weight from 0.7125 to 0.7171875 and adjust ensemble normalization from 15.7 to 15.7375.

EVIDENCE: Weight 0.71875 achieved 9,192 correct, while 0.715625 achieved 9,191 with lower cross-entropy; their midpoint is the most informative probe of the lowest weight preserving the leading correct count.

<<<<<<< SEARCH
            weight = 0.7125 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.7
=======
            weight = 0.7171875 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.7375
>>>>>>> REPLACE