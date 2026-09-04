MECHANISM: Symmetric probability-space flip ensemble

HYPOTHESIS: Averaging per-view class probabilities will exceed 9,330 correct predictions by reducing the influence of a strongly disagreeing flipped view while preserving flip symmetry; scaling the resulting log-probabilities by 1.15 should retain useful calibration.

INTENDED_EDIT: Replace evaluation-time logit averaging with an equal-weight arithmetic mixture of original and flipped softmax probabilities, returned as calibrated log-probability logits.

EVIDENCE: Equal-weight logit ensembling reached 9,330 correct, while fixed canonical weighting fell to 9,319; this motivates changing decision boundaries with a symmetric fusion rule rather than imposing a global orientation bias.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.15 * (0.5 * (logits + flipped_logits))
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            probabilities = 0.5 * (
                F.softmax(logits, dim=-1)
                + F.softmax(flipped_logits, dim=-1)
            )
            logits = 1.15 * torch.log(probabilities.clamp_min(1.0e-8))
        return logits
>>>>>>> REPLACE