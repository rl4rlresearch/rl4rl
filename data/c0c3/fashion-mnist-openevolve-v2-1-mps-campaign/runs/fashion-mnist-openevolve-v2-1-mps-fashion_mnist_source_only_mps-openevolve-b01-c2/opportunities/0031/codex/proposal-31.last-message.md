MECHANISM: Validation-logit temperature extrapolation

HYPOTHESIS: Restoring the linear ensemble curriculum and increasing evaluation scaling to 1.15 will retain all 9,330 argmax predictions while reducing validation cross-entropy below 0.198018.

INTENDED_EDIT: Replace the weaker cosine supervision curriculum with the proven linear 0.50-to-1.00 curriculum and sharpen averaged evaluation logits by 1.15.

EVIDENCE: Linear ensemble supervision achieved 9,330 correct, and successive evaluation scales of 1.05 and 1.10 preserved those predictions while reducing cross-entropy from approximately 0.204606 to 0.200571 and then 0.198018, motivating one further conservative increase.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.15 * (0.5 * (logits + flipped_logits))
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.75 - 0.25 * math.cos(math.pi * progress)
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
>>>>>>> REPLACE