MECHANISM: Accuracy-preserving post-hoc logit sharpening

HYPOTHESIS: Dividing the best-known ensemble logits by 99.5% of their exact normalization weight will retain 9,192 correct predictions and reduce validation cross-entropy below 0.23742913589477538.

INTENDED_EDIT: Restore the best-performing diagonal TTA weight and retry the timed-out 0.5% sharpening probe.

EVIDENCE: Increasing the denominator by 0.5% preserved correctness but worsened cross-entropy to 0.2380424545288086; the corresponding denominator decrease timed out without performance evidence, so retrying it is the most informative unresolved calibration test.

<<<<<<< SEARCH
                weight = 0.7161365509033203125 if is_diagonal else 1.0
                view_logits = self._forward_once(view)
                flipped_logits = self._forward_once(view.flip(-1))
                weighted_logits = weight * (view_logits + flipped_logits)
                if logit_sum is None:
                    logit_sum = weighted_logits
                else:
                    logit_sum = logit_sum + weighted_logits
        return logit_sum / 15.7290924072265625
=======
                weight = 0.7161376953125 if is_diagonal else 1.0
                view_logits = self._forward_once(view)
                flipped_logits = self._forward_once(view.flip(-1))
                weighted_logits = weight * (view_logits + flipped_logits)
                if logit_sum is None:
                    logit_sum = weighted_logits
                else:
                    logit_sum = logit_sum + weighted_logits
        return logit_sum / 15.6504560546875
>>>>>>> REPLACE