MECHANISM: Paired-view ensemble-aligned supervision

HYPOTHESIS: Training jointly on every image and its mirror, including loss on their averaged logits, will exceed 9,063 correct predictions by aligning optimization with the fixed flip-ensemble evaluation.

INTENDED_EDIT: Replace single-view cross-entropy with an equal blend of per-view and flip-averaged cross-entropy.

EVIDENCE: Evaluation-only temperature tuning has plateaued at 9,063 correct, while inference averages every translated view with its horizontal flip; the current training loss supervises only one randomly selected orientation per example.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_logits = model(torch.cat((images, images.flip(-1)), dim=0))
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    individual_loss = F.cross_entropy(
        paired_logits,
        labels.repeat(2),
        label_smoothing=0.02,
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.02,
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
>>>>>>> REPLACE