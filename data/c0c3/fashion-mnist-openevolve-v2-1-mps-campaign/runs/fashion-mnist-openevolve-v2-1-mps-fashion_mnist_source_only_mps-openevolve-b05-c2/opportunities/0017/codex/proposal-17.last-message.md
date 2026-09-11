MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.716015625 will retain 9,192 correct predictions while lowering cross-entropy below 0.2374292724609375.

INTENDED_EDIT: Bisect the diagonal-view weight between 0.715625 and 0.71640625 and adjust ensemble normalization to 15.728125.

EVIDENCE: Weight 0.71640625 achieved 9,192 correct, whereas 0.715625 achieved 9,191 with slightly lower cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
            weight = 0.71640625 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.73125
=======
            weight = 0.716015625 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.728125
>>>>>>> REPLACE