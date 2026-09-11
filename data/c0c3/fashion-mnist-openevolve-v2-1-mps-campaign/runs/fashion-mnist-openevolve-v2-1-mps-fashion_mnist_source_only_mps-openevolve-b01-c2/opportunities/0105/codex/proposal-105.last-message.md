MECHANISM: Symmetric posterior-probability flip fusion

HYPOTHESIS: Averaging class probabilities rather than logits will correct at least one borderline flip disagreement and exceed 9,330 validation-correct predictions while retaining symmetric evaluation and similar cross-entropy calibration.

INTENDED_EDIT: Replace evaluation-time geometric probability fusion with arithmetic probability fusion, retaining the verified 1.184 temperature scale.

EVIDENCE: The current equal-logit ensemble achieves 9,330 correct, and its verified temperature scaling cannot change argmax predictions; probability-space fusion is a low-cost symmetric change that can alter disputed predictions, while prior fusion experiments produced no completed contrary metrics.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (0.5 * (logits + flipped_logits))
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            view_log_probabilities = torch.stack(
                (
                    F.log_softmax(logits, dim=-1),
                    F.log_softmax(flipped_logits, dim=-1),
                ),
                dim=0,
            )
            logits = 1.184 * (
                torch.logsumexp(view_log_probabilities, dim=0)
                - math.log(2.0)
            )
        return logits
>>>>>>> REPLACE