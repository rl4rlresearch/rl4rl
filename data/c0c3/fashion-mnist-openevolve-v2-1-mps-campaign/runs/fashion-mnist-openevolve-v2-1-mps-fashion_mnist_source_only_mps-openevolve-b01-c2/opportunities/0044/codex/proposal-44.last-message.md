MECHANISM: Late hard-target annealing for symmetric flip-ensemble learning

HYPOTHESIS: Restoring the proven 5% warmup and 1.15 evaluation scale while annealing label smoothing from 0.02 to zero will exceed 9,330 correct predictions by retaining early regularization and sharpening decision boundaries late in the fixed exposure.

INTENDED_EDIT: Restore the best verified schedule and calibration, then linearly decay label smoothing throughout training without adding parameters or computation.

EVIDENCE: The 5% warmup with symmetric logit fusion reached 9,330 correct, while 2% warmup reached 9,315; alternative decision-time fusion fell to 9,319 or 9,326, so this tests a training-time margin change while preserving the best verified fusion rule.

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
) -> torch.Tensor:
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
    paired_images = torch.cat(
=======
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02 * (1.0 - progress)
    paired_images = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
=======
    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=label_smoothing,
    )
    view_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=label_smoothing,
        )
        + F.cross_entropy(
            flipped_logits,
            labels,
            label_smoothing=label_smoothing,
        )
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    warmup_steps = max(1, int(0.02 * total_steps))
=======
    warmup_steps = max(1, int(0.05 * total_steps))
>>>>>>> REPLACE