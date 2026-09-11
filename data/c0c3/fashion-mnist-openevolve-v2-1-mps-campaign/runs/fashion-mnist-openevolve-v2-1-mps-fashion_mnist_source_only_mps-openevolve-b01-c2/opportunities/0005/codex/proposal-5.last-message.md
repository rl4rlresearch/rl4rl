MECHANISM: Paired-flip deep supervision with ensemble-aligned loss

HYPOTHESIS: Supervising each horizontal orientation individually while retaining the validation-matched ensemble loss will exceed 9,261 correct predictions.

INTENDED_EDIT: Blend cross-entropy on the averaged logits with cross-entropy on each original and flipped view.

EVIDENCE: Paired-flip ensemble training reached 9,261 correct versus 9,247 for stochastic flipping, while translation fell to 9,141; this isolates stronger flip supervision without introducing a harmful augmentation.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    batch_size = images.shape[0]
    logits = 0.5 * (
        paired_logits[:batch_size] + paired_logits[batch_size:]
    )
    return F.cross_entropy(logits, labels, label_smoothing=0.02)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = 0.5 * (original_logits + flipped_logits)

    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
    return 0.5 * (ensemble_loss + view_loss)
>>>>>>> REPLACE