MECHANISM: Bounded progressive flip-orbit supervision

HYPOTHESIS: Annealing flip-ensemble weight from 60% to 90% will exceed 9,235 correct predictions by preserving the successful 75% average curriculum while avoiding the weaker 50% and 100% endpoint regimes.

INTENDED_EDIT: Replace pure flip-ensemble loss with a cosine curriculum centered at 75%, mixing individual-view and flip-averaged cross-entropy throughout training.

EVIDENCE: The 50%-to-100% curriculum achieved the best result at 9,235 correct, while static 75% reached 9,233 and static 100% fell to 9,230; narrowing the curriculum around the empirically favored weight tests whether progression helps without spending training at weaker extremes.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    return F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=0.03
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
>>>>>>> REPLACE