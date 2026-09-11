MECHANISM: Flip-orbit ensemble supervision

HYPOTHESIS: Blending per-view cross-entropy with cross-entropy on each flip pair’s averaged logits will exceed 9,224 correct predictions by directly optimizing the flip-averaged decision rule used during validation while retaining robust individual views.

INTENDED_EDIT: Replace purely independent flip-pair supervision with an equal mixture of individual-view and flip-averaged logit losses.

EVIDENCE: Complete flip-pair training improved validation from 9,155 to 9,186 correct, and the current model evaluates every translation through flip-averaged logits; directly supervising that same aggregation is the next isolated test of objective alignment.

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
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images), paired_labels, label_smoothing=0.03
    )
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
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
    return 0.5 * individual_loss + 0.5 * ensemble_loss
>>>>>>> REPLACE