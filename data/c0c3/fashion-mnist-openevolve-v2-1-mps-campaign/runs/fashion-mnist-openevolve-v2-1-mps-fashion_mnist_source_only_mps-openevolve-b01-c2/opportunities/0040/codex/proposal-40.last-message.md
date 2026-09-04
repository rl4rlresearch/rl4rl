MECHANISM: Proven flip-ensemble calibration and warmup restoration

HYPOTHESIS: Restoring the verified 5% warmup and 1.15 evaluation-logit scale will recover 9,330 correct predictions and approximately 0.196639 validation cross-entropy.

INTENDED_EDIT: Restore the best verified learning-rate schedule and sharpen the equal-weight evaluation flip ensemble without changing its argmax predictions.

EVIDENCE: Reference Design 2 used these settings with the otherwise identical implementation and achieved the best verified score, 9,330 correct with 0.196639 cross-entropy; the current 2% warmup fell to 9,315 correct.

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
    warmup_steps = max(1, int(0.02 * total_steps))
=======
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE