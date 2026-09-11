MECHANISM: Progressive flip-orbit supervision

HYPOTHESIS: Annealing from 50% to 100% flip-ensemble loss while preserving a 75% average weight will exceed 9,233 correct predictions by emphasizing individual-view learning early and matching the validation decision rule late.

INTENDED_EDIT: Replace the static 25% individual/75% ensemble mixture with a smooth cosine curriculum from equal weighting to pure ensemble supervision.

EVIDENCE: Static ensemble weights produced 9,229 correct at 50%, peaked at 9,233 at 75%, and retained 9,230 at 100%; scheduling across these viable regimes tests whether their benefits are complementary across training stages.

<<<<<<< SEARCH
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
    return 0.25 * individual_loss + 0.75 * ensemble_loss
=======
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
    ensemble_weight = 0.75 - 0.25 * math.cos(math.pi * progress)
    return (
        (1.0 - ensemble_weight) * individual_loss
        + ensemble_weight * ensemble_loss
    )
>>>>>>> REPLACE