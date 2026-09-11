MECHANISM: Probability-space mirrored-view ensembling

HYPOTHESIS: Averaging normalized class probabilities instead of raw logits will exceed 9,286 correct predictions by preventing an overconfident erroneous view from dominating the mirrored ensemble.

INTENDED_EDIT: Change only evaluation-time fusion to the log of the arithmetic mean of both views’ probabilities, retaining training, architecture, runtime, and calibrated logit scale.

EVIDENCE: Mirrored-view ensembling raised validation correct from 9,237 to 9,282, while temperature calibration cannot change the current 9,286 predictions; testing a more robust fusion rule directly targets further accuracy gains without additional forward passes.

<<<<<<< SEARCH
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 1.2112 * 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        log_probs = F.log_softmax(logits, dim=-1)
        flipped_log_probs = F.log_softmax(flipped_logits, dim=-1)
        return 1.2112 * (
            torch.logaddexp(log_probs, flipped_log_probs) - math.log(2.0)
        )
>>>>>>> REPLACE