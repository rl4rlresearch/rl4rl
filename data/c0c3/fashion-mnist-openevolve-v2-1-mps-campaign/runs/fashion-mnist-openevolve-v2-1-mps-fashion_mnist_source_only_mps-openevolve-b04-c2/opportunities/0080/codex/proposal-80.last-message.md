MECHANISM: Clean-tail stochastic translation training

HYPOTHESIS: Applying one-pixel translations to roughly 40% of examples during the first 75% of training, followed by clean-only training, will improve translation robustness and raise validation_correct above 9,279 without the degradation caused by full-strength translation test-time averaging.

INTENDED_EDIT: Add per-image cardinal one-pixel training augmentation with replicated borders and a clean final quarter, while restoring the best-verified evaluation scale of 1.17817.

EVIDENCE: Equal-weight one-pixel translation ensembling retained 9,254 correct—far closer to baseline than reflection ensembling’s 8,384—showing translations are substantially more label-preserving; learning partial invariance during training may capture their useful signal without forcing shifted predictions into every validation output.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17816 * logits
=======
        if not self.training:
            logits = 1.17817 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del step, total_steps
    return images, labels
=======
def prepare_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step < int(0.75 * total_steps):
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        variants = torch.stack(
            (
                images,
                padded[:, :, 1:-1, :-2],
                padded[:, :, 1:-1, 2:],
                padded[:, :, :-2, 1:-1],
                padded[:, :, 2:, 1:-1],
            ),
            dim=1,
        )
        choices = torch.randint(
            0, 9, (images.shape[0],), device=images.device
        ).clamp_max(4)
        images = variants[
            torch.arange(images.shape[0], device=images.device), choices
        ]
    return images, labels
>>>>>>> REPLACE