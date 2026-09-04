MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.7162109375 will retain 9,192 correct predictions while lowering cross-entropy below 0.2374292724609375.

INTENDED_EDIT: Bisect the diagonal-view weight between 0.716015625 and 0.71640625 and adjust ensemble normalization to 15.7296875.

EVIDENCE: Weight 0.71640625 achieved 9,192 correct, while 0.716015625 achieved 9,191 with lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy boundary.

<<<<<<< SEARCH
            weight = 0.7171875 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.7375
=======
            weight = 0.7162109375 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.7296875
>>>>>>> REPLACE