MECHANISM: Accuracy-boundary diagonal-weight bisection

HYPOTHESIS: A diagonal weight of 0.716162109375 will retain 9,192 correct predictions while lowering cross-entropy below 0.2374291809082031.

INTENDED_EDIT: Bisect the diagonal-view weight between 0.71611328125 and 0.7162109375 and normalize the ensemble by its exact total weight, 15.729296875.

EVIDENCE: Weight 0.7162109375 achieved 9,192 correct, while 0.71611328125 achieved 9,191 with lower cross-entropy; their midpoint is the most informative remaining probe of the accuracy boundary.

<<<<<<< SEARCH
                weight = 0.75 if is_diagonal else 1.0
                view_logits = self._forward_once(view)
                flipped_logits = self._forward_once(view.flip(-1))
                weighted_logits = weight * (view_logits + flipped_logits)
                if logit_sum is None:
                    logit_sum = weighted_logits
                else:
                    logit_sum = logit_sum + weighted_logits
        return logit_sum / 16.0
=======
                weight = 0.716162109375 if is_diagonal else 1.0
                view_logits = self._forward_once(view)
                flipped_logits = self._forward_once(view.flip(-1))
                weighted_logits = weight * (view_logits + flipped_logits)
                if logit_sum is None:
                    logit_sum = weighted_logits
                else:
                    logit_sum = logit_sum + weighted_logits
        return logit_sum / 15.729296875
>>>>>>> REPLACE