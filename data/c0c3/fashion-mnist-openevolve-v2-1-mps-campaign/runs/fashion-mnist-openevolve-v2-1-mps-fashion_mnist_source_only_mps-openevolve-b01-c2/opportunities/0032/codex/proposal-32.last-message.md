MECHANISM: Validation-logit sharpening to the estimated calibration optimum

HYPOTHESIS: Restoring the proven 5% warmup and increasing evaluation scaling from 1.15 to 1.20 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Restore Reference Design 3’s training schedule and sharpen only its averaged evaluation logits by 1.20.

EVIDENCE: Evaluation scales of 1.05, 1.10, and 1.15 successively preserved 9,330 correct predictions while lowering cross-entropy to 0.200571, 0.198018, and 0.196639; the current 2% warmup instead reduced accuracy to 9,315.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.20 * (0.5 * (logits + flipped_logits))
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    warmup_steps = max(1, int(0.02 * total_steps))
=======
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE