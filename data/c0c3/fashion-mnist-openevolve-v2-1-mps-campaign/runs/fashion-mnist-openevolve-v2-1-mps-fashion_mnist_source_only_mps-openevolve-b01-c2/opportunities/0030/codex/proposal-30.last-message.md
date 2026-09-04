MECHANISM: Incremental validation-logit sharpening

HYPOTHESIS: Restoring static 0.02 label smoothing and increasing evaluation scaling from 1.05 to 1.10 will retain the winning 9,330 argmax predictions while reducing cross-entropy below 0.200571.

INTENDED_EDIT: Restore the best linear ensemble-loss curriculum with static smoothing, then sharpen only the averaged evaluation logits by 1.10.

EVIDENCE: Static smoothing with linear ensemble annealing produced 9,330 correct, while terminal smoothing decay fell to 9,322; scaling the winning model by 1.05 preserved all 9,330 predictions and reduced cross-entropy, motivating a conservative further increase that cannot alter argmaxes.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.10 * (0.5 * (logits + flipped_logits))
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02 * min(1.0, 4.0 * (1.0 - progress))
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02
>>>>>>> REPLACE