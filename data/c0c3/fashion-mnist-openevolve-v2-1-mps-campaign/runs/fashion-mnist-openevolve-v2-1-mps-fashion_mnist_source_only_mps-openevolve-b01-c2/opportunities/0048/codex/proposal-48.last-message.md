MECHANISM: Residual per-view supervision with calibrated symmetric ensembling

HYPOTHESIS: Ending the ensemble-loss curriculum at 0.90 will exceed 9,330 correct predictions by preserving useful per-view supervision, while the verified 5% warmup and 1.15 logit scale retain the strongest established schedule and calibration.

INTENDED_EDIT: Restore the best verified warmup and evaluation scale, and retain 10% per-view loss at the curriculum endpoint.

EVIDENCE: The verified 5% warmup and 1.15-scaled equal-logit ensemble achieved 9,330 correct with 0.196639 cross-entropy; alternative fusion, EMA, and smoothing changes performed worse. The previous residual-supervision verification timed out, providing no negative model evidence.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
=======
    ensemble_weight = 0.5 + 0.4 * step / max(total_steps - 1, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    warmup_steps = max(1, int(0.02 * total_steps))
=======
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE