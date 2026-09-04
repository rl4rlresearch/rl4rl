MECHANISM: Early one-pixel translation augmentation with clean-tail fitting

HYPOTHESIS: Applying deterministic one-pixel translations during the first half of training will exceed 9,254 correct predictions by adding mild spatial regularization, while the clean second half preserves distribution-matched fitting and EMA averaging.

INTENDED_EDIT: Translate each early training batch in one of eight directions using replicated borders; leave the second half of training unchanged.

EVIDENCE: Moderate label smoothing outperformed both hard targets and 0.10 smoothing, indicating that mild regularization helps, while horizontal-reflection ensembling fell to 8,873 correct; small translations test a less destructive spatial invariance without altering the proven batch size, optimizer, or EMA tail.

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
    if step < total_steps // 2:
        offsets = (
            (0, 0), (0, 1), (0, 2), (1, 0),
            (1, 2), (2, 0), (2, 1), (2, 2),
        )
        offset_y, offset_x = offsets[step % len(offsets)]
        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        images = padded[
            ..., offset_y:offset_y + height, offset_x:offset_x + width
        ].contiguous()
    return images, labels
>>>>>>> REPLACE