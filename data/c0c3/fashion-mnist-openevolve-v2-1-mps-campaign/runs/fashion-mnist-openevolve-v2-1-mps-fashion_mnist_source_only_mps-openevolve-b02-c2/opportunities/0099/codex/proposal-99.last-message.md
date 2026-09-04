MECHANISM: Symmetric flip-consistency regularization

HYPOTHESIS: Adding a mild symmetric KL penalty between the already-computed paired-view predictions will exceed 9,322 correct predictions by encouraging flip-invariant decision boundaries without the runtime cost of additional model passes.

INTENDED_EDIT: Restore ordinary BatchNorm momentum and augment paired-view cross-entropy with a 0.10-weighted, stop-gradient symmetric KL consistency loss.

EVIDENCE: Ordinary BatchNorm reached 9,320 correct with slightly lower cross-entropy and runtime than sample-weighted BatchNorm, while arithmetic flip ensembling improved cross-entropy at the same count; repeated top-k and augmentation variants timed out, motivating a low-overhead loss that reuses existing paired logits.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    batch_fraction = images.shape[0] / BATCH_SIZE
    batch_norm_momentum = 1.0 - (1.0 - 0.1) ** batch_fraction
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum

    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
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
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    logits, flipped_logits = paired_logits.chunk(2, dim=0)

    supervised_loss = F.cross_entropy(paired_logits, paired_labels)
    log_probabilities = F.log_softmax(logits, dim=1)
    flipped_log_probabilities = F.log_softmax(flipped_logits, dim=1)
    consistency_loss = 0.5 * (
        F.kl_div(
            log_probabilities,
            flipped_log_probabilities.exp().detach(),
            reduction="batchmean",
        )
        + F.kl_div(
            flipped_log_probabilities,
            log_probabilities.exp().detach(),
            reduction="batchmean",
        )
    )
    return supervised_loss + 0.10 * consistency_loss
>>>>>>> REPLACE