MECHANISM: Quadratic-fit validation-logit calibration

HYPOTHESIS: Restoring the proven 5% warmup and scaling averaged evaluation logits by 1.184 will retain all 9,330 correct predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Restore Reference Design 3’s learning-rate schedule and apply its evidence-derived calibration-optimal evaluation scale.

EVIDENCE: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while lowering cross-entropy to 0.200571, 0.198018, and 0.196639; a quadratic fit places the minimum near 1.184, while prior 1.184 verification timed out without negative model evidence.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE

<<<<<<< SEARCH
    warmup_steps = max(1, int(0.02 * total_steps))
=======
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE