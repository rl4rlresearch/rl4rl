MECHANISM: Verified warmup restoration with calibrated symmetric flip ensembling

HYPOTHESIS: Restoring the verified 5% warmup and 1.15 evaluation scale will increase validation correctness from 9,315 to 9,330 and reduce cross-entropy toward 0.196639.

INTENDED_EDIT: Replace the regressed 2% warmup with 5% and apply the proven 1.15 temperature scale to evaluation-time averaged logits.

EVIDENCE: Reference Design 2 achieved 9,330 correct with 0.196639 cross-entropy using these exact settings, versus the current design’s 9,315 correct and 0.204467 cross-entropy.

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
    warmup_steps = max(1, int(0.02 * total_steps))
=======
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE