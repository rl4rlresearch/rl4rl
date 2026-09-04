MECHANISM: Terminal hard-label ensemble fine-tuning

HYPOTHESIS: Restoring the best batch-64 ensemble-loss curriculum and annealing label smoothing only during the final quarter will exceed 9,330 correct predictions by preserving early regularization while better matching hard-label validation late in training.

INTENDED_EDIT: Use batch size 64, linearly shift supervision from equal view/ensemble loss to pure ensemble loss, and decay label smoothing from 0.02 to zero over the final 25% of steps.

EVIDENCE: Linear ensemble-weight annealing achieved 9,330 correct versus 9,325 for static 75% weighting and 9,312 for equal weighting, showing that increasingly validation-aligned late supervision is beneficial; terminal smoothing decay extends that mechanism without removing early regularization.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

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
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02 * min(1.0, 4.0 * (1.0 - progress))

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
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
>>>>>>> REPLACE