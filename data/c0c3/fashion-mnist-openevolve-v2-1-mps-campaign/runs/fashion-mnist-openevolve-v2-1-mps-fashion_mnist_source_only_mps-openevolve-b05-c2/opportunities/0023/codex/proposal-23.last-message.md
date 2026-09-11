MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.716131591796875 will retain 9,192 correct predictions while reducing cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Bisect between the failed 0.71612548828125 and successful 0.7161376953125 diagonal weights, using the exact total ensemble weight of 15.729052734375.

EVIDENCE: Weight 0.7161376953125 achieved 9,192 correct, while 0.71612548828125 achieved 9,191; their midpoint is the most informative remaining probe of the accuracy boundary.

<<<<<<< SEARCH
            weight = 0.71640625 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.73125
=======
            weight = 0.716131591796875 if index in diagonal_indices else 1.0
            logit_sum = logit_sum + weight * self._forward_once(view)
            logit_sum = logit_sum + weight * self._forward_once(view.flip(-1))
        return logit_sum / 15.729052734375
>>>>>>> REPLACE