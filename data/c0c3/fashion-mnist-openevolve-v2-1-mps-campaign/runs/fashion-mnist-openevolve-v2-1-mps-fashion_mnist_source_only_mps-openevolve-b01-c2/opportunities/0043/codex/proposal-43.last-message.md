MECHANISM: Symmetric probability-space flip ensemble

HYPOTHESIS: Averaging per-view class probabilities will exceed 9,330 correct predictions by tolerating a strongly disagreeing view better than geometric/logit averaging, while 1.15 scaling preserves useful calibration.

INTENDED_EDIT: Replace evaluation-time equal-weight logit averaging with an equal-weight arithmetic probability mixture represented as scaled log-probabilities.

EVIDENCE: Equal-weight logit fusion achieved 9,330 correct, whereas asymmetric 0.60/0.40 fusion fell to 9,319; the prior symmetric probability-fusion verification timed out and provided no negative validation evidence.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.05 * (0.5 * (logits + flipped_logits))
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            original_log_probs = F.log_softmax(logits, dim=-1)
            flipped_log_probs = F.log_softmax(flipped_logits, dim=-1)
            mixture_log_probs = (
                torch.logaddexp(original_log_probs, flipped_log_probs)
                - math.log(2.0)
            )
            logits = 1.15 * mixture_log_probs
        return logits
>>>>>>> REPLACE